from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    # 구현하세요!
    data_list = sys.stdin.read().splitlines()

    task_num = int(data_list[0])
    task_list = data_list[1:]

    upper_bound = 1000000
    tree: SegmentTree = SegmentTree(upper_bound)
    
    ans_save = []

    for task in task_list:
        split_task = list(map(int, task.split()))
        if split_task[0] == 1:
            rank = split_task[1]
            flavor_num = tree.query(1, 1, upper_bound, rank)
            ans_save.append(flavor_num)
            tree.update(1, 1, upper_bound, flavor_num, -1)
        elif split_task[0] == 2:
            flavor_num, how_many = split_task[1], split_task[2]
            tree.update(1, 1, upper_bound, flavor_num, how_many)

    for ans in ans_save:
        print(ans)


if __name__ == "__main__":
    main()