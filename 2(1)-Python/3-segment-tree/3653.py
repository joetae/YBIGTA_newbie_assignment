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