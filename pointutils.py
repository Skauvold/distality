# Implement the IndexPointCollection class here


class IndexPoint:
    def __init__(self, indices):
        self._row_index = indices[0]
        self._col_index = indices[1]

    def __repr__(self) -> str:
        return f"({self._row_index}, {self._col_index})"

    def is_near(self, other_point):
        return (
            abs(self._row_index - other_point._row_index) <= 1
            and abs(self._col_index - other_point._col_index) <= 1
        )

    def distance_to(self, other_point, metric="euclidean"):
        if metric == "euclidean":
            return (
                (self._row_index - other_point._row_index) ** 2
                + (self._col_index - other_point._col_index) ** 2
            ) ** 0.5
        else:
            raise ValueError("Unknown metric")


class IndexPointCollection:
    def __init__(self, row_indices, col_indices):
        self.points = [IndexPoint((i, j)) for i, j in zip(row_indices, col_indices)]

    def neighbors(self, point):
        return [p for p in self.points if p.distance_to(point) > 0 and p.is_near(point)]

    def other_neighbors(self, point, previous_point):
        all_neighbors = self.neighbors(point)
        return [p for p in all_neighbors if p.distance_to(previous_point) > 0]

    def foward_neighbors(self, point, previous_point):
        all_neighbors = self.neighbors(point)
        return [
            p
            for p in all_neighbors
            if p != previous_point and p not in self.neighbors(previous_point)
        ]

    def walk_to_node(self, current_point, previous_point):
        """
        Coming from previous_point, walk in the direction of current_point until
        a dead end or a junction is reached.

        Parameters
        ----------
        current_point : IndexPoint
            The point reached by taking one step from previous_point in the given direction.
        previous_point : IndexPoint
            The point from which the current_point was reached.

        Returns
        -------
        node_point : IndexPoint
            The point where the walk ended.
        entry_point : IndexPoint
            The point from which the node_point was reached.
        node_type : str
            The type of node encountered: 'end' or 'junction'.
        distance_walked : int
            The number of steps taken from previous_point to node_point.
        segment: list
            The list of points along the path.
        """

        # Initialize the distance walked
        distance_walked = previous_point.distance_to(current_point)

        # Initialize segment
        segment = [previous_point, current_point]

        # Find forward neighbors of the current point
        forward_neighbors = self.foward_neighbors(current_point, previous_point)

        node_encountered = False

        while node_encountered == False:
            # Move to the next point
            previous_point, current_point = current_point, forward_neighbors[0]

            # Update the segment
            segment.append(current_point)

            # Update the distance walked
            distance_walked += previous_point.distance_to(current_point)

            forward_neighbors = self.foward_neighbors(current_point, previous_point)

            # Check if we have reached a node
            if len(forward_neighbors) == 0:
                node_encountered = True
                node_type = "end"
            elif len(forward_neighbors) > 1:
                node_encountered = True
                node_type = "junction"

        # Determine the node point and entry point
        node_point = current_point
        entry_point = previous_point

        return node_point, entry_point, node_type, distance_walked, segment
