def user_input():
    lines = []
    print("Describe Experiment")
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    return "\n".join(lines)
