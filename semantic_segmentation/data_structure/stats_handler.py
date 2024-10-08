class StatsHandler:
    def __init__(self, color_coding):
        self.s = {}
        self.r = {}

        self.shape_score = {"n": 0, "s": 0}

        for cls in color_coding:
            self.s[cls] = {
                "tp": 0,
                "fn": 0,
                "fp": 0,
            }

    def count(self, cls, t, val):
        self.s[cls][t] += val

    def eval(self):
        for cls in self.s:
            rec = self.s[cls]["tp"] / (self.s[cls]["tp"] + self.s[cls]["fn"] + 1e-5)
            pre = self.s[cls]["tp"] / (self.s[cls]["tp"] + self.s[cls]["fp"] + 1e-5)
            f_1 = 2 * (pre * rec) / (pre + rec + 1e-5)
            iou = self.s[cls]["tp"] / (self.s[cls]["tp"] + self.s[cls]["fn"] + self.s[cls]["fp"])
            self.r[cls] = {
                "rec": rec,
                "pre": pre,
                "f_1": f_1,
                "iou": iou,
            }

    def generate_r_string(self):
        r_string = ""
        for cls in self.r:
            r_string += "------------------\n"
            r_string += "{}-PRE: {}\n".format(cls, self.r[cls]["pre"])
            r_string += "{}-REC: {}\n".format(cls, self.r[cls]["rec"])
            r_string += "{}-F_1: {}\n".format(cls, self.r[cls]["f_1"])
            r_string += "{}-JAC: {}\n".format(cls, self.r[cls]["iou"])
            r_string += "------------------\n"

        return r_string

    def show(self):
        r_string = self.generate_r_string()
        print(r_string)

    def write_report(self, path_to_report):
        with open(path_to_report, "w") as r:
            r.write(self.generate_r_string())
