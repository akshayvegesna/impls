import os
import datetime

TEMPLATE = """


def main():
    pass

if __name__ == "__main__":
    main()

"""

def main():
    yymmdd = datetime.datetime.now().strftime("%y%m%d")
    os.mkdir(yymmdd)
    with open(yymmdd + "/main.py", "w") as f:
        f.write(TEMPLATE)

if __name__ == "__main__":
    main()


