from numpy import copy
from scipy import linalg


class History:
    """
    Data class for saving architecture search history.  
    """

    def __init__(
        self,
        model,
        architect,
        to_save=("alphas", "edges", "l2_norm", "l2_norm_from_init",),
    ):

        self.model = model
        self.architect = architect
        self.to_save = to_save
        self.dict = {}
        self.dict["epochs"] = 0

        for field in to_save:
            self.dict[field] = []

    def update_history(self, epochs):
        self.dict["epochs"] = epochs
        for field in self.to_save:
            if field == "alphas":
                values = {
                    ct: self.architect.alphas[ct].data.cpu().numpy()
                    for ct in self.architect.cell_types
                }
                self.dict["alphas"].append(values)
            elif field == "edges":
                values = {
                    ct: self.architect.edges[ct].data.cpu().numpy()
                    for ct in self.architect.cell_types
                }
                self.dict["edges"].append(values)
            elif field == "graph_laplacians":
                values = {
                    ct: copy(self.architect.graph_laplacians[ct])
                    for ct in self.architect.cell_types
                }
                self.dict["graph_laplacians"].append(values)
            elif field == "l2_norm":
                self.dict["l2_norm"].append(self.model.compute_norm(from_init=False))
            elif field == "l2_norm_from_init":
                self.dict["l2_norm_from_init"].append(
                    self.model.compute_norm(from_init=True)
                )
            else:
                # Assume field is a property of architect of type numpy array
                self.dict[field].append(copy(getattr(self.architect, field)))

    def log_vars(self, epoch, writer):
        for field in self.to_save:
            print(field)
            last_v = self.dict[field][-1]
            if field == "alphas":
                for ct in self.architect.cell_types:
                    writer.add_image(
                        "{}_{}".format(ct, field), last_v[ct], epoch, dataformats="HW"
                    )
            elif field == "graph_laplacians":
                for ct in self.architect.cell_types:
                    writer.add_image(
                        "{}_{}".format(ct, field), last_v[ct], epoch, dataformats="HW"
                    )
                    norm1 = linalg.norm(last_v[ct], ord=1)
                    norm2 = linalg.norm(last_v[ct], ord=2)
                    writer.add_scalar("{}_graph_norm1".format(ct), norm1, epoch)
                    writer.add_scalar("{}_graph_norm2".format(ct), norm2, epoch)
