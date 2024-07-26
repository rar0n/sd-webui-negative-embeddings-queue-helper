import os
import copy
import random
import math
import re

import gradio as gr

from modules import sd_samplers, errors, scripts, images, sd_models
from modules.processing import Processed, process_images
from modules.shared import state, cmd_opts, opts
from pathlib import Path

embeddings_dir = Path(cmd_opts.embeddings_dir).resolve()

# Sorting key for case insensitiv and alphanumeric sorting
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def allowed_path(path):
    original_path = Path(path)
    resolved_path = original_path.resolve()
    is_allowed = original_path.is_relative_to(embeddings_dir) or resolved_path.is_relative_to(embeddings_dir)
    return is_allowed

def get_base_path(is_use_custom_path, custom_path):
    return embeddings_dir.joinpath(custom_path) if is_use_custom_path else embeddings_dir

def is_directory_contain_embedding(path):
    try:
        if allowed_path(path):
            embedding_files = [f.name for f in os.scandir(path) if f.is_file(follow_symlinks=True) and (f.name.endswith('.pt') or f.name.endswith('.safetensors'))]
            return len(embedding_files) > 0
    except Exception as e:
        print(f"Error checking directory {path}: {e}")
    return False

def get_directories(base_path, include_root=True):
    directories = ["/"] if include_root else []
    try:
        if allowed_path(base_path):
            for entry in os.scandir(base_path):
                if entry.is_dir(follow_symlinks=True):
                    full_path = entry.path
                    if is_directory_contain_embedding(full_path):
                        directories.append(entry.name)

                    nested_directories = get_directories(full_path, include_root=False)
                    directories.extend([os.path.join(entry.name, d) for d in nested_directories])
    except Exception as e:
        print(f"Error getting directories in {base_path}: {e}")
    #return sorted(directories, key=str.lower)
    return sorted(directories, key=natural_sort_key)

def get_embeddings(base_path, directories):
    all_embeddings = []
    for directory in directories:
        directory_path = base_path if directory == "/" else base_path.joinpath(directory)
        if not allowed_path(directory_path):
            continue
        try:
            embedding_files = [f.name for f in os.scandir(directory_path) if f.is_file(follow_symlinks=True) and (f.name.endswith('.pt') or f.name.endswith('.safetensors'))]
            all_embeddings.extend([os.path.splitext(f)[0] for f in embedding_files])
        except Exception as e:
            print(f"Error getting embeddings in {directory_path}: {e}")
    #return sorted(all_embeddings, key=str.lower)
    return sorted(all_embeddings, key=natural_sort_key)

class Script(scripts.Script):
    def title(self):
        return "Queue selected Embeddings (batch) - Negative Prompt"

    def ui(self, is_img2img):
        def update_dirs(is_use_custom_path, custom_path):
            base_path = get_base_path(is_use_custom_path, custom_path)
            dirs = get_directories(base_path)
            return gr.CheckboxGroup.update(choices=dirs, value=[])

        def show_dir_textbox(enabled, custom_path):
            all_dirs = get_directories(embeddings_dir.joinpath(custom_path) if enabled else embeddings_dir)
            return gr.Textbox.update(visible=enabled), gr.CheckboxGroup.update(choices=all_dirs, value=[])

        def update_embeddings(current_selected, is_use_custom_path, custom_path, directories):
            base_path = get_base_path(is_use_custom_path, custom_path)
            all_embeddings = get_embeddings(base_path, directories)
            visible = len(all_embeddings) > 0
            new_values = [embedding for embedding in all_embeddings if embedding in current_selected]
            return gr.CheckboxGroup.update(choices=all_embeddings, value=new_values, visible=visible), gr.Button.update(visible=visible), gr.Button.update(visible=visible)

        def select_all_dirs(is_use_custom_path, custom_path):
            base_path = get_base_path(is_use_custom_path, custom_path)
            all_dirs = get_directories(base_path)
            return gr.CheckboxGroup.update(value=all_dirs)

        def deselect_all_dirs():
            return gr.CheckboxGroup.update(value=[])

        def select_all_embeddings(is_use_custom_path, custom_path, directories):
            base_path = get_base_path(is_use_custom_path, custom_path)
            all_embeddings = get_embeddings(base_path, directories)
            return gr.CheckboxGroup.update(value=all_embeddings)

        def deselect_all_embeddings():
            return gr.CheckboxGroup.update(value=[])

        def toggle_row_number(checked):
            return gr.Number.update(visible=checked), gr.Checkbox.update(visible=checked)

        def toggle_auto_row_number(checked):
            return gr.Number.update(interactive=not checked)

        with gr.Column():
            base_dir_checkbox = gr.Checkbox(label="Use Custom Embeddings path", value=False, elem_id=self.elem_id("base_dir_checkbox"))
            base_dir_textbox = gr.Textbox(label="Embeddings directory", placeholder="Relative path under Embeddings directory. Use --embeddings-dir to set Embeddings directory at WebUI startup.", visible=False, elem_id=self.elem_id("base_dir_textbox"))
            base_dir = base_dir_textbox.value if base_dir_checkbox.value else embeddings_dir
            all_dirs = get_directories(base_dir)

            directory_checkboxes = gr.CheckboxGroup(label="Select Directory", choices=all_dirs, value=["/"], elem_id=self.elem_id("directory_checkboxes"))

            with gr.Row():
                select_all_dirs_button = gr.Button("All")
                deselect_all_dirs_button = gr.Button("Clear")

            startup_embeddings = get_embeddings(base_dir, directory_checkboxes.value)
            
            embedding_checkboxes = gr.CheckboxGroup(label="Embeddings", choices=startup_embeddings, value=startup_embeddings, visible=len(startup_embeddings)>0, elem_id=self.elem_id("embedding_checkboxes"))

            with gr.Row():
                select_all_embeddings_button = gr.Button("All", visible=len(startup_embeddings)>0)
                deselect_all_embeddings_button = gr.Button("Clear", visible=len(startup_embeddings)>0)

            with gr.Row():
                checkbox_iterate = gr.Checkbox(label="Use consecutive seed", value=False, elem_id=self.elem_id("checkbox_iterate"))
                checkbox_iterate_batch = gr.Checkbox(label="Use same random seed", value=False, elem_id=self.elem_id("checkbox_iterate_batch"))
            
            with gr.Row(equal_height=True):
                with gr.Column():
                    checkbox_save_grid = gr.Checkbox(label="Save grid image", value=True, elem_id=self.elem_id("checkbox_save_grid"))
                    checkbox_auto_row_number = gr.Checkbox(label="Auto row number", value=True, elem_id=self.elem_id("checkbox_auto_row_number"))
                grid_row_number = gr.Number(label="Grid row number", value=1, interactive=False, elem_id=self.elem_id("grid_row_number"))

            base_dir_checkbox.change(fn=show_dir_textbox, inputs=[base_dir_checkbox, base_dir_textbox], outputs=[base_dir_textbox, directory_checkboxes])
            base_dir_textbox.change(fn=update_dirs, inputs=[base_dir_checkbox, base_dir_textbox], outputs=[directory_checkboxes])
            directory_checkboxes.change(fn=update_embeddings, inputs=[embedding_checkboxes, base_dir_checkbox, base_dir_textbox, directory_checkboxes], outputs=[embedding_checkboxes, select_all_embeddings_button, deselect_all_embeddings_button])
            select_all_embeddings_button.click(fn=select_all_embeddings, inputs=[base_dir_checkbox, base_dir_textbox, directory_checkboxes], outputs=embedding_checkboxes)
            deselect_all_embeddings_button.click(fn=deselect_all_embeddings, inputs=None, outputs=embedding_checkboxes)
            select_all_dirs_button.click(fn=select_all_dirs, inputs=[base_dir_checkbox, base_dir_textbox], outputs=directory_checkboxes)
            deselect_all_dirs_button.click(fn=deselect_all_dirs, inputs=None, outputs=directory_checkboxes)
            checkbox_save_grid.change(fn=toggle_row_number, inputs=checkbox_save_grid, outputs=[grid_row_number, checkbox_auto_row_number])
            checkbox_auto_row_number.change(fn=toggle_auto_row_number, inputs=[checkbox_auto_row_number], outputs=grid_row_number)

        return [base_dir_checkbox, base_dir_textbox, directory_checkboxes, embedding_checkboxes, checkbox_iterate, checkbox_iterate_batch, checkbox_save_grid, checkbox_auto_row_number, grid_row_number]

    def run(self, p, is_use_custom_path, custom_path, directories, selected_embeddings, checkbox_iterate, checkbox_iterate_batch, is_save_grid, is_auto_row_number, row_number):
        if len(selected_embeddings) == 0:
            return process_images(p)

        p.do_not_save_grid = True  # disable default grid image

        job_count = 0
        jobs = []

        base_path = get_base_path(is_use_custom_path, custom_path)
        all_embeddings = get_embeddings(base_path, directories)

        for embedding_name in all_embeddings:
            if embedding_name not in selected_embeddings:
                continue

            args = {}
            # Apply the embedding to the negative prompt instead of the positive prompt
            args["negative_prompt"] = f"{embedding_name}, " + (p.negative_prompt or "")

            job_count += args.get("n_iter", p.n_iter)

            jobs.append(args)

        if (checkbox_iterate or checkbox_iterate_batch) and p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.job_count = job_count

        result_images = []
        all_prompts = []
        all_negative_prompts = []
        infotexts = []
        for args in jobs:
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            proc = process_images(copy_p)
            result_images += proc.images

            if checkbox_iterate:
                p.seed = p.seed + (p.batch_size * p.n_iter)
            all_prompts += proc.all_prompts
            all_negative_prompts += proc.all_negative_prompts
            infotexts += proc.infotexts

        if is_save_grid and len(result_images) > 1:
            if is_auto_row_number:
                # get a 4:3 rectangular width
                row_number = round(3.0 * math.sqrt(len(result_images)/12.0))
            else:
                row_number = int(row_number)

            grid_image = images.image_grid(result_images, rows=row_number)
            result_images.insert(0, grid_image)
            all_prompts.insert(0, "")
            all_negative_prompts.insert(0, "")
            infotexts.insert(0, "")

        return Processed(p, result_images, p.seed, "", all_prompts=all_prompts, all_negative_prompts=all_negative_prompts, infotexts=infotexts)
