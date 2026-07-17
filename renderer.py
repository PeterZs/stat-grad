# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

import nvdiffrast.torch as dr

glctx = dr.RasterizeCudaContext()


def transform_pos(mtx, pos):
    posw = torch.cat([pos, torch.ones([pos.shape[0], pos.shape[1], 1]).cuda()], axis=2)
    return torch.matmul(posw, mtx.permute(0, 2, 1))


class NormalRenderer(nn.Module):
    def __init__(self, image_size, obj_filename, eyehole_path, mouthhole_path):
        super(NormalRenderer, self).__init__()
        
        self.image_size = image_size
        
        # Load mesh
        _, faces, _ = load_obj(obj_filename)
        faces = faces.verts_idx[None, ...]

        source_eyehole = torch.from_numpy(np.load(eyehole_path))
        source_mouthhole = torch.from_numpy(np.load(mouthhole_path))
        
        source_hole = torch.cat([source_eyehole, source_mouthhole])
        self.register_buffer('faces', faces)
        self.register_buffer('pos_idx', faces[0].int())
        self.register_buffer('hole_idx', source_hole)
        
        self.enable_culling = False

    def forward(self, vertices_world, verts_noneck, r_mvps):
        """
        Args:
            vertices_world: [B, N, 3] - vertices in world space for rendering
            verts_noneck: [B, N, 3] - vertices for normal calculation
            r_mvps: [B, 4, 4] - MVP matrix
            
        Returns:
            rendered_normals: [B, 3, H, W] - rendered normal maps
        """
        B = vertices_world.shape[0]
        faces = self.faces.expand(B, -1, -1)
        
        # Compute normals from mesh
        meshes_world = Meshes(verts=vertices_world.float(), faces=faces.long())
        normals = meshes_world.verts_normals_packed().reshape(B, -1, 3)
        
        # Transform vertices to clip space
        pos_clips = transform_pos(r_mvps, vertices_world).float()
        
        # Rasterize
        rast_out, rast_out_db = dr.rasterize(
            glctx, pos_clips, self.pos_idx,
            resolution=[self.image_size, self.image_size]
        )
        
        # Interpolate normals
        rendered_normals = dr.interpolate(normals, rast_out, self.pos_idx)[0]
        
        if self.enable_culling:
            # Screen-space backface culling
            face_normal_z = rendered_normals[..., 2:3]  # z component
            mask = (face_normal_z > 0).float()
            rendered_normals = rendered_normals * mask
        
        rendered_normals = rendered_normals.permute(0, 3, 1, 2)  # [B, 3, H, W]
        mask = rendered_normals != 0
        
        # Render eyehole mask
        # Create vertex colors: 1 for eyehole vertices, 0 for others
        vertex_colors = torch.zeros_like(vertices_world)[...,0:1].type_as(vertices_world)
        vertex_colors[:, self.hole_idx] = 1.0
        vertex_colors = vertex_colors.contiguous()
        
        # Interpolate vertex colors to get eyehole mask
        hole_mask = dr.interpolate(vertex_colors, rast_out, self.pos_idx)[0]
        hole_mask = dr.antialias(
            hole_mask,
            rast_out,
            pos_clips,
            self.pos_idx
        )
        hole_mask = hole_mask.permute(0, 3, 1, 2)  # [B, 1, H, W]

        skin_colors = torch.ones_like(vertices_world)[...,0:1].type_as(vertices_world).contiguous()
        skin_mask = dr.interpolate(skin_colors, rast_out, self.pos_idx)[0]
        skin_mask = dr.antialias(
            skin_mask,
            rast_out,
            pos_clips,
            self.pos_idx
        )
        skin_mask = skin_mask.permute(0, 3, 1, 2)  # [B, 1, H, W]

        
        outputs = {
            'normal_images': rendered_normals,
            'mask': mask,
            'hole_mask': hole_mask,
            'skin_mask': skin_mask,
        }
        return outputs
