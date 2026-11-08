"""Private CPU research context: one analyzed source, bounded reusable buffers."""
import numpy as np
import onnxruntime as ort


def session(path, *, optimized=True, threads=1):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimized
                                        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


class CachedContext:
    def __init__(self, split, *, optimized=True, threads=1, query_threads=None):
        self.analysis = session(split/'analysis.onnx', optimized=optimized, threads=threads)
        self.query = session(split/'query.onnx', optimized=optimized,
                             threads=threads if query_threads is None else query_threads)
        self.image = np.empty((1, 3, 256, 256), np.float32)
        self.content = np.empty((1, 4), np.float32)
        self.boxes = np.empty((1, 128, 4), np.float32)
        self.outputs = {}
        for output in self.analysis.get_outputs():
            if output.type != 'tensor(float)' or not all(isinstance(v, int) and v > 0 for v in output.shape):
                raise ValueError('fixed positive FP32 analysis output required')
            if np.prod(output.shape) > 256*256*64:
                raise ValueError('analysis output exceeds research context bound')
            self.outputs[output.name] = np.empty(output.shape, np.float32)
        self.scores = np.empty((1, 128), np.float32)
        self.analysis_binding = self.analysis.io_binding()
        self.analysis_binding.bind_cpu_input('image', self.image)
        for name, value in self.outputs.items():
            self.analysis_binding.bind_output(name, 'cpu', 0, np.float32, value.shape, value.ctypes.data)
        self.query_binding = self.query.io_binding()
        self.query_binding.bind_cpu_input('boxes', self.boxes)
        self.query_binding.bind_cpu_input('content', self.content)
        for input in self.query.get_inputs():
            if input.name not in ['boxes', 'content']:
                self.query_binding.bind_cpu_input(input.name, self.outputs[input.name])
        self.query_binding.bind_output('crop_scores', 'cpu', 0, np.float32, self.scores.shape, self.scores.ctypes.data)
        self.generation, self.token = 0, None

    def analyze(self, image, content, source_identity):
        self.token = None
        if (image.shape != self.image.shape or content.shape != self.content.shape or
                not np.isfinite(image).all() or not np.isfinite(content).all() or
                not (0 <= content[0, 0] < content[0, 2] <= 1 and
                     0 <= content[0, 1] < content[0, 3] <= 1)):
            raise ValueError('bounded source tensor/content required')
        np.copyto(self.image, image)
        np.copyto(self.content, content)
        self.analysis.run_with_iobinding(self.analysis_binding)
        if not all(np.isfinite(value).all() for value in self.outputs.values()):
            raise ValueError('nonfinite source analysis')
        self.generation += 1
        self.token = (self.generation, source_identity)
        return self.token

    def score(self, boxes, token):
        if token is None or token != self.token:
            raise ValueError('stale or foreign source analysis token')
        # Fixed slots include zero-area padding. The downstream integer crop
        # planner validates actual candidates; scoring never certifies faces.
        if (boxes.shape != self.boxes.shape or not np.isfinite(boxes).all() or
                not ((boxes >= 0).all() and (boxes <= 1).all() and
                     (boxes[..., :2] <= boxes[..., 2:]).all())):
            raise ValueError('bounded normalized candidate slots required')
        np.copyto(self.boxes, boxes)
        self.query.run_with_iobinding(self.query_binding)
        if not np.isfinite(self.scores).all():
            raise ValueError('nonfinite cached crop scores')
        return self.scores

    def retained_tensor_bytes(self):
        return sum(value.nbytes for value in [self.image, self.content, self.boxes,
                                               self.scores, *self.outputs.values()])


def prepared(auxiliary, index):
    valid = auxiliary['packed']['valid_rect'][index].numpy()
    x0, y0, x1, y1 = np.rint(valid*256).astype(int)
    image = np.zeros((1, 3, 256, 256), np.float32)
    mean = np.array([.485, .456, .406], np.float32)[:, None, None]
    std = np.array([.229, .224, .225], np.float32)[:, None, None]
    image[0, :, y0:y1, x0:x1] = ((auxiliary['image'][index].numpy()-mean)/std)[:, y0:y1, x0:x1]
    return image, valid[None]
