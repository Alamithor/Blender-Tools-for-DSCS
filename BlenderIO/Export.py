import bpy
import bmesh
from collections import Counter
import numpy as np
import itertools
import os
import shutil
from bpy_extras.io_utils import ExportHelper
from bpy_extras.image_utils import load_image
from bpy_extras.object_utils import object_data_add
from mathutils import Vector
from ..CollatedData.ToReadWrites import generate_files_from_intermediate_format
from ..CollatedData.IntermediateFormat import IntermediateFormat
from ..FileReaders.GeomReader.ShaderUniforms import shader_uniforms_from_names, shader_textures, shader_uniforms_vp_fp_from_names


class ExportDSCSBase:
    bl_label = 'Digimon Story: Cyber Sleuth (.name, .skel, .geom)'
    bl_options = {'REGISTER'}
    filename_ext = ".name"

    def export_file(self, context, filepath, platform, copy_shaders=True):
        model_data = IntermediateFormat()
        export_folder = os.path.join(*os.path.split(filepath)[:-1])
        export_images_folder = os.path.join(export_folder, 'images')
        os.makedirs(export_images_folder, exist_ok=True)
        export_shaders_folder = os.path.join(export_folder, 'shaders')
        if copy_shaders:
            os.makedirs(export_shaders_folder, exist_ok=True)

        used_materials = []
        used_textures = []
        # Grab the parent object
        parent_obj = self.get_model_to_export()
        self.export_skeleton(parent_obj, model_data)
        self.export_meshes(parent_obj, model_data, used_materials)
        self.export_materials(model_data, used_materials, used_textures, export_shaders_folder)
        self.export_textures(used_textures, model_data, export_images_folder)

        model_data.unknown_data['material names'] = [material.name for material in model_data.materials]
        # Top-level unknown data
        model_data.unknown_data['unknown_cam_data_1'] = parent_obj.get('unknown_cam_data_1', [])
        model_data.unknown_data['unknown_cam_data_2'] = parent_obj.get('unknown_cam_data_2', [])
        model_data.unknown_data['unknown_footer_data'] = parent_obj.get('unknown_footer_data', b'')
        generate_files_from_intermediate_format(filepath, model_data, platform)

    def get_model_to_export(self):
        try:
            parent_obj = bpy.context.selected_objects[0]

            sel_obj = None
            while parent_obj is not None:
                sel_obj = parent_obj
                parent_obj = sel_obj.parent
            parent_obj = sel_obj
            return parent_obj
        except Exception as e:
            raise Exception("No object selected. Ensure you have selected some part of the model you wish to export in "
                            "Object Mode before attempting to export.") from e

    def export_skeleton(self, parent_obj, model_data):
        model_armature = parent_obj.children[0]
        bone_name_list = [bone.name for bone in model_armature.data.bones]
        for i, bone in enumerate(model_armature.data.bones):
            name = bone.name
            parent_bone = bone.parent
            parent_id = bone_name_list.index(parent_bone.name) if parent_bone is not None else -1

            model_data.skeleton.bone_names.append(name)
            model_data.skeleton.bone_relations.append([i, parent_id])
            model_data.skeleton.inverse_bind_pose_matrices.append(np.linalg.inv(np.array(bone.matrix_local)))

        # Get the unknown data
        model_data.skeleton.unknown_data['unknown_0x0C'] = model_armature.get('unknown_0x0C', 0)
        model_data.skeleton.unknown_data['unknown_data_1'] = model_armature.get('unknown_data_1', [])
        model_data.skeleton.unknown_data['unknown_data_2'] = model_armature.get('unknown_data_2', [])
        model_data.skeleton.unknown_data['unknown_data_3'] = model_armature.get('unknown_data_3', [])
        model_data.skeleton.unknown_data['unknown_data_4'] = model_armature.get('unknown_data_4', [])

    def export_meshes(self, parent_obj, model_data, used_materials):
        mat_names = []
        for i, mesh_obj in enumerate(parent_obj.children[0].children):
            md = model_data.new_mesh()
            mesh = mesh_obj.data

            link_loops = self.generate_link_loops(mesh)
            face_link_loops = self.generate_face_link_loops(mesh)
            export_verts, export_faces, vgroup_verts, vgroup_wgts = self.split_verts_by_uv(mesh_obj, link_loops, face_link_loops, model_data)

            md.vertices = export_verts
            for j, face in enumerate(export_faces):
                assert len(face) == 3, f"Polygon {j} is not a triangle."
                md.add_polygon(face)

            for group in mesh_obj.vertex_groups:
                bone_name = group.name
                bone_id = model_data.skeleton.bone_names.index(bone_name)
                md.add_vertex_group(bone_id, vgroup_verts.get(bone_id, []), vgroup_wgts.get(bone_id, []))

            matname = mesh.materials[0].name
            if matname not in mat_names:
                md.material_id = len(used_materials)
                used_materials.append(mesh.materials[0])
            else:
                md.material_id = mat_names.index(matname)

            md.unknown_data['unknown_0x31'] = mesh_obj.get('unknown_0x31', 1)
            md.unknown_data['unknown_0x34'] = mesh_obj.get('unknown_0x34', 0)
            md.unknown_data['unknown_0x36'] = mesh_obj.get('unknown_0x36', 0)
            md.unknown_data['unknown_0x4C'] = mesh_obj.get('unknown_0x4C', 0)

    def generate_link_loops(self, mesh):
        link_loops = {}
        for loop in mesh.loops:
            if loop.vertex_index not in link_loops:
                link_loops[loop.vertex_index] = []
            link_loops[loop.vertex_index].append(loop.index)
        return link_loops

    def generate_face_link_loops(self, mesh):
        face_link_loops = {}
        for face in mesh.polygons:
            for loop_idx in face.loop_indices:
                face_link_loops[loop_idx] = face.index
        return face_link_loops

    def split_verts_by_uv(self, mesh_obj, link_loops, face_link_loops, model_data):
        mesh = mesh_obj.data
        exported_vertices = []
        vgroup_verts = {}
        vgroup_wgts = {}
        faces = [{l: mesh.loops[l].vertex_index for l in f.loop_indices} for f in mesh.polygons]

        if 'UV3Map' in mesh.uv_layers:
            map_ids = ['UVMap', 'UV2Map', 'UV3Map']
        elif 'UV2Map' in mesh.uv_layers:
            map_ids = ['UVMap', 'UV2Map']
        elif 'UVMap' in mesh.uv_layers:
            map_ids = ['UVMap']
        else:
            map_ids = []
        generating_function = lambda lidx: tuple([tuple(mesh.uv_layers[map_id].data.values()[lidx].uv) for map_id in map_ids])
        for vert_idx, linked_loops in link_loops.items():
            vertex = mesh.vertices[vert_idx]
            loop_uvs = [generating_function(ll) for ll in linked_loops]
            unique_values = list(set(loop_uvs))
            for unique_value in unique_values:
                loops_with_this_value = [linked_loops[i] for i, x in enumerate(loop_uvs) if x == unique_value]

                group_bone_ids = [get_bone_id(mesh_obj, model_data.skeleton.bone_names, grp) for grp in vertex.groups]
                group_bone_ids = None if len(group_bone_ids) == 0 else group_bone_ids
                group_weights = [grp.weight for grp in vertex.groups]
                group_weights = None if len(group_weights) == 0 else group_weights

                vert = {'Position': vertex.co,
                        'Normal': vertex.normal,
                        **{key: value for key, value in zip(['UV', 'UV2', 'UV3'], unique_value)},
                        'WeightedBoneID': [grp.group for grp in vertex.groups],
                        'BoneWeight': group_weights}
                # Grab the tangents, bitangents, colours for each UV-split vertex?

                n_verts = len(exported_vertices)
                exported_vertices.append(vert)

                for l in loops_with_this_value:
                    face_idx = face_link_loops[l]
                    faces[face_idx][l] = n_verts

                if group_bone_ids is not None:
                    for group_bone_id, weight in zip(group_bone_ids, group_weights):
                        if group_bone_id not in vgroup_verts:
                            vgroup_verts[group_bone_id] = []
                            vgroup_wgts[group_bone_id] = []
                        vgroup_verts[group_bone_id].append(n_verts)
                        vgroup_wgts[group_bone_id].append(weight)

        faces = [list(face_verts.values()) for face_verts in faces]

        return exported_vertices, faces, vgroup_verts, vgroup_wgts

    def export_materials(self, model_data, used_materials, used_textures, export_shaders_folder):
        tex_names = []
        for bmat in used_materials:
            material = model_data.new_material()
            node_tree = bmat.node_tree
            material.name = bmat.name
            material.unknown_data['unknown_0x00'] = bmat.get('unknown_0x00', 0)
            material.unknown_data['unknown_0x02'] = bmat.get('unknown_0x02', 0)
            material.shader_hex = bmat.get('shader_hex',
                                           '088100c1_00880111_00000000_00058000')  # maybe use 00000000_00000000_00000000_00000000 instead
            material.unknown_data['unknown_0x16'] = bmat.get('unknown_0x16', 1)

            if 'shaders_folder' in bmat:
                for shader_filename in os.listdir(bmat['shaders_folder']):
                    if shader_filename[:35] == material.shader_hex:
                        try:
                            shutil.copy2(os.path.join(bmat['shaders_folder'], shader_filename),
                                         os.path.join(export_shaders_folder, shader_filename))
                        except shutil.SameFileError:
                            continue

            # Export Textures
            node_names = [node.name for node in node_tree.nodes]
            for nm in shader_textures:
                if nm in node_names:
                    texture = node_tree.nodes[nm].image

                    # Construct the texture index
                    texname = texture.name
                    if texname in tex_names:
                        tex_idx = tex_names.index(texname)
                    else:
                        tex_idx = len(used_textures)

                    # Construct the additional, unknown data
                    extra_data = bmat.get(nm)
                    if extra_data is None:
                        extra_data = [0, 0]
                    else:
                        extra_data = extra_data[1:]  # Chop off the texture idx

                    material.shader_uniforms[nm] = [tex_idx, *extra_data]
                    used_textures.append(node_tree.nodes[nm].image)

            # Export the material components
            for key in shader_uniforms_vp_fp_from_names.keys():
                if bmat.get(key) is not None:
                    material.shader_uniforms[key] = bmat.get(key)
            material.unknown_data['unknown_material_components'] = {}
            for key in ['160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '172']:
                if bmat.get(key) is not None:
                    material.unknown_data['unknown_material_components'][key] = bmat.get(key)

    def export_textures(self, used_textures, model_data, export_images_folder):
        used_texture_names = [tex.name for tex in used_textures]
        used_texture_paths = [tex.filepath for tex in used_textures]
        for texture, texture_path in zip(used_texture_names, used_texture_paths):
            tex = model_data.new_texture()
            tex.name = texture
            try:
                shutil.copy2(texture_path,
                             os.path.join(export_images_folder, texture + ".img"))
            except shutil.SameFileError:
                continue
            except FileNotFoundError:
                print(texture_path, "not found.")
                continue

    def execute_func(self, context, filepath, platform):
        filepath, file_extension = os.path.splitext(filepath)
        assert any([file_extension == ext for ext in
                    ('.name', '.skel', '.geom')]), f"Extension is {file_extension}: Not a name, skel or geom file!"
        self.export_file(context, filepath, platform)

        return {'FINISHED'}


class ExportDSCSPC(ExportDSCSBase, bpy.types.Operator, ExportHelper):
    bl_idname = 'export_file.export_dscs_pc'

    def execute(self, context):
        return super().execute_func(context, self.filepath, 'PC')


class ExportDSCSPS4(ExportDSCSBase, bpy.types.Operator, ExportHelper):
    bl_idname = 'export_file.export_dscs_ps4'

    def execute(self, context):
        return super().execute_func(context, self.filepath, 'PS4')


def get_bone_id(mesh_obj, bone_names, grp):
    group_idx = grp.group
    bone_name = mesh_obj.vertex_groups[group_idx].name
    bone_id = bone_names.index(bone_name)
    return bone_id
