
import hashlib
                   
class Vertex():
    def __init__(self, label="0", neighbors=[]):
        """
            Parameters:
            ==========
            .label (str): The label of this vertex drawn from the set {1,2,...,n} 
            .valence(int): number of edges incident on a vertex V_i
            .indicant_edge(tuple): an edge (i,j) such that i=v or j=v.
            .neighbors: neighbors of this vertex (as labels).

            
            Test
            ====
                b0 = Vertex("0")
                b1 = Vertex("1")
                b2 = Vertex("2")
                b3 = Vertex("3")
                b0.update_neighbor(b1)
                b0.update_neighbor(b2)
                b2.update_neighbor(b3)
                print(b0)
                print(b1)
                print(b2)
                print(b3)

                Prints: Vertex: 0 | Neighbors: ['1', '2']
                        Vertex: 1 | Neighbors: ['0']
                        Vertex: 2 | Neighbors: ['0', '3']
                        Vertex: 3 | Neighbors: ['2']

                Multiple neighbors test
                -----------------------
                neighs_2 = [Vertex(f"{i}") for i in range(3, 9)]
                b2.update_neighbors(neighs_2)
                print(b2)

                Prints: Vertex: 2 | Neighbors: ['3', '4', '5', '6', '7', '8']

        """
        self.label = label
        self.neighbors = []
        self.inidicant_edge = []

    def update_neighbor(self, neigh):
        if isinstance(neigh, list):
            for neigh_single in neigh:
                self.update_neighbor(neigh_single)
            return
        assert isinstance(neigh, Vertex), "Neighbor must be a Vertex member function."
        # assert neigh not in self.neighbors, "No repeated neighbors allowed."
        # assert neigh!=self, "Cannot assign a vertex as its own neighbor"

        if neigh in self.neighbors or neigh==self:
            return self.neighbors 

        self.neighbors.append(neigh)

        # this neighbor must be a neighbor of this parent
        neigh.neighbors.append(self) 

    @property
    def valence(self):
        """
            By how much has the number of edges incident
            on v changed?

            Parameter
            =========
            delta: integer (could be positive or negative).

            It is positive if the number of egdes increases at a time t.
            It is negative if the number of egdes decreases at a time t.
        """
        return len(self.neighbors)

    
    def update_inidicant_edges(self, edges):
        """
            Update the number of edges (i,j) of the graph for which either
            i=j or j=v.
        """        
        
    def __hash__(self):
        # hash method to distinguish agents from one another    
        return int(hashlib.md5(str(self.label).encode('utf-8')).hexdigest(),16)

    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False

    def __repr__(self):
        parent=f"Vertex: {self.label} | "
        children="Neighbors: 0" if not self.neighbors \
                else f"Neighbors: {sorted([x.label for x in self.neighbors])}"
        return parent + children  


b0 = Vertex("0")
b1 = Vertex("1")
b2 = Vertex("2")
b3 = Vertex("3")
b0.update_neighbor(b1)
b0.update_neighbor(b2)
b2.update_neighbor(b3)
print(b0)
print(b1)
print(b2)
print(b3)