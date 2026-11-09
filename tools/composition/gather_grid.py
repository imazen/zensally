"""Fixed student-query GridSample lowering, qualified numerically after export."""
import io

import onnx
import torch
from torch import nn
from artifact_identity import sha


class BorderGridGather(nn.Module):
    """Bilinear border sampling with align_corners=True for finite coordinates."""
    def forward(self, table, grid):
        batch, channels, height, width = table.shape
        x = ((grid[..., 0]+1)*((width-1)*.5)).clamp(0, width-1)
        y = ((grid[..., 1]+1)*((height-1)*.5)).clamp(0, height-1)
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = (x0+1).clamp_max(width-1), (y0+1).clamp_max(height-1)
        dx, dy = (x-x0)[:, None], (y-y0)[:, None]
        flat = table.flatten(2)

        def take(xx, yy):
            indices = (yy*width+xx).flatten(1)[:, None].expand(-1, channels, -1)
            return flat.gather(2, indices).reshape(batch, channels, grid.shape[1], grid.shape[2])

        top = take(x0, y0)*(1-dx)+take(x1, y0)*dx
        bottom = take(x0, y1)*(1-dx)+take(x1, y1)*dx
        return top*(1-dy)+bottom*dy


def lower_student_grid_samples(path):
    if sha(path) not in ['56ee763e9ac9615fe3a1ab3ff564a03080deb0f94e058e329dd656a0d5304b53',
                         '2070ca2d7b01846d8ff27933434345dc58d1af6072c4540a39dcbc0e9bdd789b']:
        raise ValueError('only the two frozen original student graphs may be lowered')
    model = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
    shapes = {v.name:[d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    replacement, initializers, changed = [], [], 0
    for node in model.graph.node:
        if node.op_type != 'GridSample':
            replacement.append(node)
            continue
        attributes = {a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
        if attributes != dict(align_corners=1, mode=b'bilinear', padding_mode=b'border'):
            raise ValueError('only the qualified finite border/align-corners student sampler is supported')
        # Shape inference leaves symbolic Pad/reshape extents unknown (zero).
        # Exact original graph hashes above bind the registered two scales/order.
        table_shape = [1, 16, 33, 33] if changed == 0 else [1, 16, 17, 17]
        grid_shape = [1, 2816, 4, 2]
        for actual, fixed in [(shapes[node.input[0]], table_shape), (shapes[node.input[1]], grid_shape)]:
            if len(actual) != len(fixed) or any(a not in [0, b] for a,b in zip(actual, fixed)):
                raise ValueError('fixed student sampler dimensions differ: '+str((actual, fixed)))
        buffer = io.BytesIO()
        torch.onnx.export(BorderGridGather().eval(), (torch.zeros(table_shape), torch.zeros(grid_shape)),
            buffer, input_names=['table', 'grid'], output_names=['values'],
            opset_version=18, dynamo=False, external_data=False)
        subgraph = onnx.load_model_from_string(buffer.getvalue()).graph
        prefix = 'student_gather_'+str(changed)+'/'
        public = dict(table=node.input[0], grid=node.input[1], values=node.output[0])

        def name(value):
            return public.get(value, prefix+value) if value else value

        for part in subgraph.node:
            copy = onnx.NodeProto()
            copy.CopyFrom(part)
            copy.name = prefix+copy.name
            for index, value in enumerate(copy.input):
                copy.input[index] = name(value)
            for index, value in enumerate(copy.output):
                copy.output[index] = name(value)
            replacement.append(copy)
        for value in subgraph.initializer:
            copy = onnx.TensorProto()
            copy.CopyFrom(value)
            copy.name = name(copy.name)
            initializers.append(copy)
        changed += 1
    if changed != 2:
        raise ValueError('exactly two fixed student samplers required')
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    model.graph.initializer.extend(initializers)
    # Old inferred intermediates are regenerated for the new graph.
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))
