class ResultSys:
    def __init__(self, id, imageSys, listAnswer=None):
        self.id = id
        self.imageSys = imageSys
        self.listAnswer = listAnswer if listAnswer is not None else []