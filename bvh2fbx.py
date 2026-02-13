import bpy
import os
import argparse
import sys

FBX_MAPPING = {
    "Y Bot": "assets/fbx/Y Bot.fbx"

}

def purge_all_data(motion_length):
    """clear all blender cache data"""
    # for obj in bpy.data.objects:
    #     bpy.data.objects.remove(obj, do_unlink=True)
    # for action in bpy.data.actions:
    #     bpy.data.actions.remove(action)
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()
    bpy.data.objects['Camera'].select_set(True)
    bpy.ops.object.delete()

    for block_type in [bpy.data.meshes, bpy.data.armatures, bpy.data.actions, 
                      bpy.data.materials, bpy.data.textures, bpy.data.images]:
        for block in block_type:
            if block.users == 0:
                try:
                    block_type.remove(block)
                except:
                    pass
    
    if hasattr(bpy.context.scene, 'keemap_settings'):
        settings = bpy.context.scene.keemap_settings
        settings.source_rig_name = ""
        settings.bone_mapping_file = ""

    
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = motion_length
    bpy.ops.outliner.orphans_purge(do_recursive=True)



def check_and_install_addon():

    addon_path = "assets/KeeMap.Rig.Transfer.Addon.0.0.9.zip"
    addon_name = "KeeMapAnimRetarget"

    if addon_name not in bpy.context.preferences.addons:
        print(f"addon {addon_name} not install, installing now...")
        
        bpy.ops.preferences.addon_install(filepath=addon_path)
        
        bpy.ops.preferences.addon_enable(module=addon_name)

        bpy.ops.wm.save_userpref()

        print(f"addon {addon_name} installed")
    else:
        print(f"addon {addon_name} already installed")


def bvh2fbx(bvh_path, fbx_choice, output_path, motion_length=196):
    """
    The following code is a translated script from the motion retarget tutorial: 
    [YouTube Link](https://www.youtube.com/watch?v=EG-VCMkVpxg).
    """
    
    fbx_path = f"assets/fbx/{fbx_choice}.fbx"
    mapping_file = 'assets/mapping.json'
    
    check_and_install_addon()
    purge_all_data(motion_length)
    
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    bpy.ops.import_anim.bvh(filepath=bvh_path)

    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')  

    bpy.ops.object.select_all(action='DESELECT')

    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']

    if not armatures:
        raise Exception("There is no Armature in the scene.")
    else:
        for arm in armatures:
            arm.select_set(True) 
    
    bpy.context.view_layer.objects.active = armatures[-1]
    bpy.ops.object.mode_set(mode='POSE')
   
    bpy.data.scenes["Scene"].keemap_settings.bone_mapping_file = mapping_file
    bpy.ops.wm.keemap_read_file()

    rig_name = os.path.splitext(os.path.basename(bvh_path))[0]
    bpy.data.scenes["Scene"].keemap_settings.source_rig_name = rig_name
    bpy.data.scenes["Scene"].keemap_settings.number_of_frames_to_apply = motion_length

    armature = bpy.data.objects.get("Armature")
    desti_prefix = bpy.data.scenes["Scene"].keemap_bone_mapping_list[0].DestinationBoneName.split(':')[0]
    rename_desti_prefix = armature.pose.bones[0].name.split(':')[0]

    # Different Mixamo characters may have variations in bone naming, 
    # so it's necessary to standardize the names.
    for i in range(len(bpy.data.scenes["Scene"].keemap_bone_mapping_list)):
        bpy.data.scenes["Scene"].keemap_bone_mapping_list[i].DestinationBoneName = bpy.data.scenes["Scene"].keemap_bone_mapping_list[i].DestinationBoneName.replace(desti_prefix, rename_desti_prefix)

    # perform retarget
    bpy.ops.wm.perform_animation_transfer()

    # save
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')

    armature = bpy.data.objects.get("Armature")

    if not armature:
        raise Exception("There is no Armature in the scene.")

    armature.select_set(True)
    for child in armature.children_recursive:
        child.select_set(True)

    bpy.context.scene.render.fps = 20
    bpy.context.scene.render.fps_base = 1.0

    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        object_types={'ARMATURE', 'MESH'},
        bake_anim=True,
        path_mode='COPY',           
        embed_textures=True,
        bake_anim_use_all_bones=True,
        add_leaf_bones=False,
        use_armature_deform_only=True,
        bake_space_transform=True
    )

    glb_output_path = output_path.replace('.fbx', '.glb')

    bpy.ops.export_scene.gltf(
        filepath=glb_output_path,
        export_format='GLB',
        use_selection=True,
        export_apply=True
    )

if __name__ == "__main__":
    try:
        separator_idx = sys.argv.index("--")
        script_args = sys.argv[separator_idx + 1:]  # 取出 '--' 之后的部分
    except ValueError:
        script_args = sys.argv
    parser = argparse.ArgumentParser(description="Process BVH to FBX conversion.")
    parser.add_argument('--bvh_path', type=str, required=True, help="Path to the BVH file.")
    parser.add_argument('--fbx_choice', type=str, required=True, help="Choice of FBX model.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to output the FBX file.")
    parser.add_argument('--motion_length', type=int, default=196, help="Length of the motion (default: 196).")
    args = parser.parse_args(script_args)
    bvh2fbx(args.bvh_path, args.fbx_choice, args.output_path, args.motion_length)

    # bvh_path = "assets/bvh/sample_repeat0.bvh"
    # fbx_choice = "Y Bot"
    # output_path = "res.fbx"
    # bvh2fbx(bvh_path, fbx_choice, output_path, motion_length=180)