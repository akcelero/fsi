# Search methods

import search

ab = search.GPSProblem('A', 'B', search.romania)


pattern = "Expanded {} : Result {}"
result =search.breadth_first_graph_search(ab)
print(pattern.format(result[0], result[1].path()))
result = search.depth_first_graph_search(ab)
print(pattern.format(result[0], result[1].path()))
#  print search.iterative_deepening_search(ab).path()
#  print search.depth_limited_search(ab).path()

result = search.priority_choosing_search(ab)
print(pattern.format(result[0], result[1].path()))

result = search.priority_choosing_with_h_search(ab)
print(pattern.format(result[0], result[1].path()))

#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
