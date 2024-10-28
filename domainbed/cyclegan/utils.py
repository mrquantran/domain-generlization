def get_sources(dataset: str, targets: list):
    if dataset == "PACS":
        domains = ["art_painting", "cartoon", "photo", "sketch"]
        domain_names = {0: "art_painting", 1: "cartoon", 2: "photo", 3: "sketch"}
        for i in targets:
            domains.remove(domain_names[i])
        return domains

    elif dataset == "OfficeHome":
        domains = ["Art", "Clipart", "Product", "Real_World"]
        domain_names = {0: "Art", 1: "Clipart", 2: "Product", 3: "Real_World"}
        for i in targets:
            domains.remove(domain_names[i])
        return domains

    elif dataset == "DomainNet":
        domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        domain_names = {
            0: "clipart",
            1: "infograph",
            2: "painting",
            3: "quickdraw",
            4: "real",
            5: "sketch",
        }
        for i in targets:
            domains.remove(domain_names[i])
        return domains

    elif dataset == "TerraIncognita":
        domains = ["L100", "L38", "L43", "L46"]
        domain_names = {0: "L100", 1: "L38", 2: "L43", 3: "L46"}
        for i in targets:
            domains.remove(domain_names[i])
        return domains

    elif dataset == "VLCS":
        domains = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]
        domain_names = {0: "Caltech101", 1: "LabelMe", 2: "SUN09", 3: "VOC2007"}
        for i in targets:
            domains.remove(domain_names[i])
        return domains

    else:
        raise NotImplementedError("Dataset not implemented yet for GANs")
