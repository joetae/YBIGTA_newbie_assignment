from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    # 구현하세요!
    def __init__(self, size: int) -> None:
        self.size = size
        self.tree = [0] * (4*size)
    
    def query(self, node: int, start: int, end: int, rank: int) -> int:
        if start == end:
            return start
        
        mid = (start + end) // 2
        left = self.tree[node*2]
        if left >= rank:
            return self.query(node*2, start, mid, rank)
        else:
            return self.query(node*2+1, mid+1, end, rank-left)
        

    def query_range(self, node: int, start: int, end: int, left: int, right: int) -> int:
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_sum = self.query_range(node*2, start, mid, left, right)
        right_sum = self.query_range(node*2 + 1, mid + 1, end, left, right)
        return left_sum + right_sum

    def update(self, node: int, start: int, end: int, index: int, diff: int) -> None:
        if index < start or end < index:
            return

        self.tree[node] = self.tree[node] + diff
        
        if start != end:
            mid = (start + end) // 2
            self.update(node*2, start, mid, index, diff)
            self.update(node*2 + 1, mid + 1, end, index, diff)
    


import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    # 구현하세요!
    data_list = sys.stdin.read().splitlines()

    test_case = int(data_list[0])
    ans_save = []

    for i in range(test_case):
        # n: 영화 수, m: 행동 수
        n, m = map(int, data_list[1+2*i].split())
        movie_save = list(map(int, data_list[2 + 2*i].split()))

        upper_bound = n + m
        tree: SegmentTree = SegmentTree(upper_bound)
        movie_location = {}

        for j in range(1, n + 1):
            # 초기 위치는 트리의 맨 오른쪽부터 채워넣고 시작
            movie_location[j] = m + j
            tree.update(1, 1, upper_bound, movie_location[j], 1)

        ans_temp = []
        current_top = m

        for movie in movie_save:
            # 누적합을 통해 위에 쌓인 영화 개수 계산
            current_location = movie_location[movie]
            how_many_movie = tree.query_range(1, 1, upper_bound, 1, current_location - 1)
            ans_temp.append(how_many_movie)

            # 영화 빼주기
            tree.update(1, 1, upper_bound, current_location, -1)

            # 빼준 영화를 맨 위로
            current_top = current_top - 1
            movie_location[movie] = current_top
            tree.update(1, 1, upper_bound, current_top, 1)

        ans_save.append(" ".join(map(str, ans_temp)))

    for ans in ans_save:
        print(ans)



if __name__ == "__main__":
    main()