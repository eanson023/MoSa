
import os
from os.path import join as pjoin

from functools import partial
print("Starting")
import torch
print(f"Is CUDA available: {torch.cuda.is_available()}")
# print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
import numpy as np
import gradio as gr
import uuid
import shutil

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from utils.plot_script import plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain

from visualization.joints2bvh import Joint2BVHConvertor

from text2bvh import load_vq_model, load_trans_model, text2bvh

IS_HF_SPACE = "SPACE_ID" in os.environ
static_source_proj_path = '../static_source/'

clip_version = 'ViT-B/32'


WEBSITE = """
<div class="embed_hidden">
<h1 style='text-align: center'> MoSa: Motion Generation with Scalable Autoregressive Modeling </h1>
<h2 style='text-align: center'>
<a href="" target="_blank"><nobr>Anonymous authors</nobr></a> &emsp;
</h2>
<h2 style='text-align: center'>
<nobr>Submitted to ICCV 2025</nobr>
</h2>
</div>
"""

WEBSITE_bottom = """
<div class="embed_hidden">
<p>
We use the <a href="https://github.com/nkeeline/Keemap-Blender-Rig-ReTargeting-Addon" target="_blank" rel="noopener">KeeMap</a> Blender plugin to complete the joint retargeting, and use <a href="https://github.com/BabylonJS/Babylon.js" target="_blank" rel="noopener">Babylon.js</a> for rendering. Thank them! 
</p>
</div>
"""

EXAMPLES = [
   "A person is running on a treadmill.", "The person takes 4 steps backwards.", 
#    "A person jumps up and then lands.", 
   "The person was pushed but did not fall.", 
   "The person does a salsa dance.", "A figure streches it hands and arms above its head.",
   "This person kicks with his right leg then jabs several times.",
   "A person stands for few seconds and picks up his arms and shakes them.",
   "A person walks in a clockwise circle and stops where he began.",
   "A man bends down and picks something up with his right hand.",
   "a person moves ahead a few steps quickly.",
   "a person walks forward and is pushed forward.",
   "this man is bent forward and walks slowly around.",
#    "A person walks with a limp, their left leg gets injured.",
   "A person repeatedly blocks their face with their right arm.",
#    "The person holds his left foot with his left hand, puts his right foot up and left hand up too.",
   "The person holds their left foot with their left hand, lifting both their left foot and left hand up.",
#    "A person stands, crosses left leg in front of the right, lowering themselves until they are sitting, both hands on the floor before standing and uncrossing legs.",
   "The person stands, crosses their left leg in front of the right, lowers themselves until they are sitting with both hands on the floor, and then stands back up, uncrossing their legs.",
   "a person brings up their right hand from their side to touch their nose, drops their arm back down, and takes two steps backwards.",
#    "The man walked forward, spun right on one foot and walked back to his original position.",
   "A man is walking forward then steps over an object then continues walking forward.",
   "a person dances in celebration, thrusting their arms in the air and moving side to side.",
   "a person does a cartwheel and then walks in a circle to the right."

]

# Show closest text in the training


# css to make videos look nice
# var(--block-border-color); TODO
CSS = """
.generate_video {
    position: relative;
    margin-left: auto;
    margin-right: auto;
    box-shadow: var(--block-shadow);
    border-width: var(--block-border-width);
    border-color: #000000;
    border-radius: var(--block-radius);
    background: var(--block-background-fill);
    line-height: var(--line-sm);
}
}
"""


DEFAULT_TEXT = "A person is "


if IS_HF_SPACE:
    prompt_path = "/data/stats/Prompts.text"
    if not os.path.exists("/data/checkpoints/t2m"):
        os.system("bash prepare/download_models_demo.sh")
    if not os.path.exists("checkpoints/t2m"):
        os.system("ln -s /data/checkpoints/checkpoints checkpoints")
    if not os.path.exists("/data/stats"):
        os.makedirs("/data/stats")
        with open(prompt_path, 'w') as f:
            pass
else:
    prompt_path = "./Prompts.text"
    if not os.path.exists("./stats"):
        os.makedirs("./stats")
        with open(prompt_path, 'w') as f:
            pass

Total_Calls = 0
def update_total_calls(prompt_path):
    global Total_Calls
    Total_Calls_offset = 0 ## init number from visit, 03/08
    with open(prompt_path, 'r') as f:
        Total_Calls = len(f.readlines()) + Total_Calls_offset
    print("Prompts Num:", Total_Calls)

### Load Stats ###

##########################
######Preparing demo######
##########################
uid = ''

parser = EvalT2MOptions()
opt = parser.parse()
# fixseed(opt.seed)

opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

dim_pose = 263
opt.nb_joints = 22
# --------------------------- HumanML3D --------------------
# out_dir = pjoin(opt.check)
root_dir = pjoin(opt.checkpoints_dir, 't2m', 't2m_pkeep_rope_ffsize768_bs64_milestone100_200')

model_opt_path = pjoin(root_dir, 'opt.txt')
model_opt = get_opt(model_opt_path, device=opt.device)


#######################
######Loading SVQ######
#######################
vq_opt_path = pjoin(opt.checkpoints_dir, 't2m', model_opt.vq_name, 'opt.txt')
vq_opt = get_opt(vq_opt_path, device=opt.device)
vq_opt.dim_pose = dim_pose
t2m_vq_model = load_vq_model(vq_opt)

model_opt.scales = vq_opt.scales
model_opt.nb_code_st = vq_opt.nb_code_st
model_opt.nb_code_ed = vq_opt.nb_code_ed
model_opt.code_dim = vq_opt.code_dim

#################################
######Loading Transformer######
#################################
t2m_transformer = load_trans_model(model_opt, opt, 'net_best_fid.tar')

t2m_transformer.eval()
t2m_vq_model.eval()

t2m_transformer.to(opt.device)
t2m_vq_model.to(opt.device) 

##### ---- Dataloader ---- #####
t2m_mean = np.load(pjoin(opt.checkpoints_dir, 't2m', model_opt.vq_name, 'meta', 'mean.npy'))
t2m_std = np.load(pjoin(opt.checkpoints_dir, 't2m', model_opt.vq_name, 'meta', 'std.npy'))
def t2m_inv_transform(data):
    return data * t2m_std + t2m_mean



# --------------------------- Motion-X --------------------
# out_dir = pjoin(opt.check)
root_dir = pjoin(opt.checkpoints_dir, 'motionx', 't2m_pkeep_rope_ffsize768_bs64_milestone100_200')

model_opt_path = pjoin(root_dir, 'opt.txt')
model_opt = get_opt(model_opt_path, device=opt.device)


#######################
######Loading SVQ######
#######################
vq_opt_path = pjoin(opt.checkpoints_dir, 'motionx', model_opt.vq_name, 'opt.txt')
vq_opt = get_opt(vq_opt_path, device=opt.device)
vq_opt.dim_pose = dim_pose
motionx_vq_model = load_vq_model(vq_opt)

model_opt.scales = vq_opt.scales
model_opt.nb_code_st = vq_opt.nb_code_st
model_opt.nb_code_ed = vq_opt.nb_code_ed
model_opt.code_dim = vq_opt.code_dim

#################################
######Loading Transformer######
#################################
motionx_transformer = load_trans_model(model_opt, opt, 'net_best_fid.tar')

motionx_transformer.eval()
motionx_vq_model.eval()

motionx_transformer.to(opt.device)
motionx_vq_model.to(opt.device) 

##### ---- Dataloader ---- #####
motionx_mean = np.load(pjoin(opt.checkpoints_dir, 'motionx', model_opt.vq_name, 'meta', 'mean.npy'))
motionx_std = np.load(pjoin(opt.checkpoints_dir, 'motionx', model_opt.vq_name, 'meta', 'std.npy'))
def motionx_inv_transform(data):
    return data * t2m_std + t2m_mean

# ------------------------------ Done ---------------------------

converter = Joint2BVHConvertor()
if IS_HF_SPACE:
    cached_dir = '/tmp/gradio'
else:
    cached_dir = static_source_proj_path + 'cached'

def clear_cache_on_startup(cached_dir):
    if os.path.exists(cached_dir):
        print(f'delete {cached_dir}')
        shutil.rmtree(cached_dir)
    os.makedirs(cached_dir, exist_ok=True)

clear_cache_on_startup(cached_dir)

def setup_blender():

    blender_version = '4.0.0'
    blender_url = "https://ftp.nluug.nl/pub/graphics/blender/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz"
    blender_filename = blender_url.split("/")[-1]  # Extracts correct filename

    blender_path = f"/tmp/blender"
    extracted_path = f"{blender_path}/{blender_version}"

    if not os.path.exists(extracted_path):  
        print(f"Downloading Blender {blender_version} from {blender_url}...")
        proxy = ''
        if not IS_HF_SPACE:
            proxy = '-e "https_proxy=http://127.0.0.1:7890"'
        os.system(f"wget {proxy} {blender_url} -O /tmp/{blender_filename}")

        print("Extracting Blender...")
        os.system(f"mkdir -p {extracted_path}")
        os.system(f"tar -xf /tmp/{blender_filename} -C {extracted_path} --strip-components=1")

    if not os.path.exists(f"{extracted_path}/blender"):  # Check if extraction succeeded
        raise Exception("Error: Blender was not extracted correctly.")

    return extracted_path


# Function to handle rendering
def render_scene(bvh_path, fbx_choice, fbx_path, motion_length):
    # Setup Blender version
    blender_extracted_path = setup_blender()

    blender_command = f"{blender_extracted_path}/blender --background --python bvh2fbx.py -- --bvh_path {bvh_path} --fbx_choice '{fbx_choice}' --output_path {fbx_path} --motion_length {motion_length}"

    print("Running Blender...")
    print(blender_command)
    os.system(blender_command)


# HTML component
def get_video_html(bvh_path, fbx_path, glb_path):
    # class="wrap default svelte-gjihhp hide"
    # <div class="contour_video" style="position: absolute; padding: 10px;">
    # width="{width}" height="{height}"
    print(bvh_path)
    print(fbx_path)
    print(glb_path)
    if IS_HF_SPACE:
        bvh_url = f'gradio_api/file={bvh_path}'
        fbx_download_url = f'gradio_api/file={fbx_path}'
        glb_download_url = f'gradio_api/file={glb_path}'
        glb_url = f'https://mosa-web-static-source.hf.space/app.html?glb=https://mosa-web-mosa.hf.space/{glb_download_url}'
    else:
        bvh_url = f'http://localhost:5000/{bvh_path[len(static_source_proj_path):]}'
        fbx_download_url = f'http://localhost:5000/{fbx_path[len(static_source_proj_path):]}'
        glb_download_url = f'http://localhost:5000/{glb_path[len(static_source_proj_path):]}'
        glb_url = f'http://localhost:5000/app.html?glb={glb_path[len(static_source_proj_path):]}'
    video_html = f"""
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
    <a href="{bvh_url}" download="sample.bvh" style="padding: 10px 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">
        <b>BVH Download</b>
    </a>
    <a href="{fbx_download_url}" download="sample.fbx" style="padding: 10px 20px; background: #2196F3; color: white; text-decoration: none; border-radius: 5px;">
        <b>FBX Download</b>
    </a>
    <a href="{glb_download_url}" download="sample.fbx" style="padding: 10px 20px; background: #FF5722; color: white; text-decoration: none; border-radius: 5px;">
        <b>GLB Download</b>
    </a>
    </div>
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <iframe src='{glb_url}' width='80%' height='500px'></iframe>
    </div>
    """
    return video_html

def get_video_html_mp4(bvh_path, mp4_path, width=400, height=400):
    # class="wrap default svelte-gjihhp hide"
    # <div class="contour_video" style="position: absolute; padding: 10px;">
    # width="{width}" height="{height}"
    print(bvh_path)
    print(mp4_path)
    if IS_HF_SPACE:
        bvh_url = f'gradio_api/file={bvh_path}'
        mp4_url = f'gradio_api/file={mp4_path}'
    else:
        bvh_url = f'http://localhost:5000/{bvh_path[len(static_source_proj_path):]}'
        mp4_url = f'http://localhost:5000/{mp4_path[len(static_source_proj_path):]}'
    video_html = f"""
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
    <a href="{bvh_url}" download="sample.bvh" style="padding: 10px 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">
        <b>BVH Download</b>
    </a>
    </div>
    <video class="generate_video" width="{width}" height="{height}" style="center" preload="auto" muted playsinline onpause="this.load()"
    autoplay loop disablepictureinpicture id="0">
    <source src="{mp4_url}" type="video/mp4">
    Your browser does not support the video tag.
    </video>
    """
    return video_html


def generate_component(generate_function, retarget_finction, text, role, step, up2motion_len, motion_len, dataset, visual_type, seed, cfg, topp, topk):
    if text == DEFAULT_TEXT or text == "" or text is None:
        return [None for _ in range(1)]
    uid = uuid.uuid1()
    try:
        motion_len = max(0, min(int(float(motion_len) * 20), 196))
    except:
        motion_len = 196
    print(text)
    with open(prompt_path, 'a') as f:
        f.write(text+'\n')
    update_total_calls(prompt_path)
    if dataset == 'HumanML3D':
        transformer = t2m_transformer
        vq_model = t2m_vq_model
        inv_transform = t2m_inv_transform
    elif dataset == 'Motion-X':
        transformer = motionx_transformer
        vq_model = motionx_vq_model
        inv_transform = motionx_inv_transform
    else:
        raise NotImplementedError("")
    step = step-1
    bvh_path, joint, motion_len = generate_function(transformer, vq_model, converter, cached_dir, uid, text, step, up2motion_len, inv_transform, motion_length=motion_len, seed=seed, cond_scale=cfg, top_k=topk, top_p=topp)
    if visual_type == "Joint":
        mp4_path = bvh_path.replace('.bvh', '.mp4')
        plot_3d_motion(mp4_path, t2m_kinematic_chain, joint, text, fps=20)
        return [get_video_html_mp4(bvh_path, mp4_path)]
    else:
        fbx_path = bvh_path.replace('.bvh', '.fbx')
        glb_path = bvh_path.replace('.bvh', '.glb')
        retarget_finction(bvh_path, role, fbx_path, motion_len)
        return [get_video_html(bvh_path, fbx_path, glb_path)]


# LOADING

# DEMO
theme = gr.themes.Default(primary_hue="blue", secondary_hue="gray")
generate_and_show = partial(generate_component, text2bvh, render_scene)

gallery = [
    ('assets/img/Y Bot.png', 'Y Bot'),
    ('assets/img/X Bot.png', 'X Bot'),
    ('assets/img/Jackie.png', 'Jackie'),
    ('assets/img/Michelle.png', 'Michelle'),
    ('assets/img/Exo Gray.png', 'Exo Gray'),
    ('assets/img/Amy.png', 'Amy'),
    ('assets/img/Jolleen.png', 'Jolleen'),
    ('assets/img/Kaya.png', 'Kaya'),
]
   

with gr.Blocks(css=CSS, theme=theme) as demo:
    gr.Markdown(WEBSITE)
    videos = []

    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(
                show_label=True,
                label="Text prompt",
                value=DEFAULT_TEXT,
            )
        
            role = gr.Gallery(
                label="Charactors",
                value=gallery,
                show_label=True, 
                preview=False, 
                allow_preview=False, 
                elem_id="gallery",
                columns=[len(gallery)], 
                rows=[1],
                height='100px',
                selected_index = 0,
            )

            selected_role = gr.Textbox(gallery[0][1], visible=False)
            
            def get_selected_role(evt: gr.SelectData):
                return gallery[evt.index][1] 
            
            # When a character is selected, update the hidden textbox
            role.select(
                fn=get_selected_role,
                outputs=selected_role,
            )
            with gr.Row():
                step = gr.Slider(1, 10, value=10, scale=7, step=1, label="Scales (inference steps)", 
                        info="From 1 to 10, it shows the generation from coarse to fine.")
                up2motion_len = gr.Checkbox(False, label="Upsampling to motion length", 
                                            scale=3, interactive=True,
                                            info="Upsampling to motion length")
            
            motion_len = gr.Slider(
                        1, 10,
                        show_label=True,
                        label="Motion length (<10s)",
                        value=10,
                        info="Specify the motion length.",
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    dataset = gr.Radio(
                        ["HumanML3D", "Motion-X"],
                        label="Dataset",
                        value="HumanML3D",
                        info="The different pre-trained weights will be used.",
                    )
                with gr.Column(scale=1):
                    visual_type = gr.Radio(
                        ["Joint", "Charactor"],
                        label="Visualization type",
                        value="Joint",
                        info="The Character visualization will consume much time.",
                        # interactive=False
                    )
                
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=9999,
                        step=1,
                        value=2025,
                    )
                
                    cfg = gr.Slider(
                        label="Classifier-free Guidance (CFG)",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=4.0,  
                    )
                
                with gr.Row():
                    top_p = gr.Slider(
                        label="Top-p",
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        value=0.9,
                    )
                
                    top_k = gr.Slider(
                        label="Top-k",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.85,  
                    )

            gen_btn = gr.Button("Generate", variant="primary")
            clear = gr.Button("Clear", variant="secondary")
            gr.Markdown(
                        f"""
                            
                        """
                    )
            

        with gr.Column(scale=2):

            examples = gr.Examples(
                examples=[[x, None, None] for x in EXAMPLES],
                inputs=[text],
                examples_per_page=20,
                run_on_click=False,
                cache_examples=False,
                fn=generate_and_show,
                outputs=[],
            )

    i = -1
    # should indent
    for _ in range(1):
        with gr.Row():
            for _ in range(1):
                i += 1
                video = gr.HTML()
                videos.append(video)
    gr.Markdown(WEBSITE_bottom)
    # connect the examples to the output
    # a bit hacky
    examples.outputs = videos

    def load_example(example_id):
        processed_example = examples.non_none_processed_examples[example_id]
        return gr.utils.resolve_singleton(processed_example)

    examples.dataset.click(
        load_example,
        inputs=[examples.dataset],
        outputs=examples.inputs_with_examples,  # type: ignore
        show_progress=False,
        postprocess=False,
        queue=False,
    ).then(fn=generate_and_show, inputs=examples.inputs + [selected_role, step, up2motion_len, motion_len, dataset, visual_type, seed, cfg, top_p, top_k], outputs=videos)

    
    gen_btn.click(
        fn=generate_and_show,
        inputs=[text, selected_role, step, up2motion_len, motion_len, dataset, visual_type, seed, cfg, top_p, top_k],
        outputs=videos,
    )
    text.submit(
        fn=generate_and_show,
        inputs=[text, selected_role, step, up2motion_len, motion_len, dataset, visual_type, seed, cfg, top_p, top_k],
        outputs=videos,
    )

    def clear_videos():
        def get_defaults(text):
            return [
                text,
                gallery[0][1],  # selected_role
                10,  # step
                False,  # up2motion_len
                10,  # motion_len
                "HumanML3D",  # dataset
                "Jint",
                2025,  # seed
                4.0,  # cfg
                0.9,  # top_p
                0.85  # top_k
            ]
        return [None for x in range(1)] + get_defaults(DEFAULT_TEXT)

    clear.click(fn=clear_videos, 
                outputs=videos + [text, step, up2motion_len, motion_len, dataset, visual_type, seed, cfg, top_p, top_k])

demo.queue(default_concurrency_limit=5)
demo.launch()
