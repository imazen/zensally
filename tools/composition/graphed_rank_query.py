"""Private fixed-shape query replay; original model/training state stays separate."""
import copy

import torch


class GraphedRankQuery:
    def __init__(self, model):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.graph = None
        self.active = False

    @torch.no_grad()
    def prepare(self, model):
        if self.active:
            raise ValueError('query context already active')
        for target, source in zip(self.model.ranker.scorer.parameters(), model.ranker.scorer.parameters(), strict=True):
            target.copy_(source)
        self.active = True
        self.first_query = True

    @torch.no_grad()
    def scores(self, features, boxes, source_aspect, target_aspect, mode, content):
        if not self.active:
            raise ValueError('query context requires an explicit evaluation lifetime')
        values = (*features, boxes, source_aspect, target_aspect, mode, content)
        if self.graph is None:
            self.inputs = tuple(v.clone() for v in values)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.model.scores(self.inputs[:2], *self.inputs[2:])
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self.model.scores(self.inputs[:2], *self.inputs[2:])
        for index, (target, source) in enumerate(zip(self.inputs, values, strict=True)):
            if (target.shape != source.shape or target.dtype != source.dtype or target.device != source.device):
                raise ValueError('query replay shape/dtype/device contract differs')
            if index >= 2 or self.first_query:
                target.copy_(source)
        if not self.first_query and self.feature_identity != tuple((v.data_ptr(), v._version) for v in features):
            raise ValueError('features changed inside one evaluation lifetime')
        self.feature_identity = tuple((v.data_ptr(), v._version) for v in features)
        self.first_query = False
        self.graph.replay()  # Capture alone does not initialize outputs.
        return self.output

    def finish(self):
        if not self.active:
            raise ValueError('query context is not active')
        self.active = False


def evaluate_with_graph(context, model, evaluate, *args, verify=None):
    """Retain original evaluation, masks, reductions and finite-value checks."""
    original = model.scores
    context.prepare(model)

    def scores(*values):
        result = context.scores(*values)
        if verify is not None:
            verify(result, original(*values))
        return result

    model.scores = scores
    try:
        return evaluate(model, *args)
    finally:
        del model.scores
        context.finish()
