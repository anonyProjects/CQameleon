import ast
import numpy as np
from sklearn.cluster import DBSCAN
from Levenshtein import distance  
from tqdm import tqdm

class CodeVisitor(ast.NodeTransformer):
    def __init__(self, words):
        self.words = words
        self.defined_vars = {}  
        self.idx = 0

    def visit_FunctionDef(self, node):
        
        node.name = f"{node.name}_{self.words[0]}"
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id not in self.defined_vars:
                    
                    self.defined_vars[target.id] = self.words[self.idx % len(self.words)]
                    self.idx += 1
                target.id = f"{target.id}_{self.defined_vars[target.id]}"
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        
        if node.id in self.defined_vars:
            node.id = f"{node.id}_{self.defined_vars[node.id]}"
        return node

def compute_distance_matrix(words):
    n = len(words)
    distance_matrix = np.zeros((n, n))
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            dist = distance(words[i], words[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix

def cluster_words_unsupervised(words, eps=1.5, min_samples=2):
    
    distance_matrix = compute_distance_matrix(words)

    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(distance_matrix)
    return labels

def poison(targets,query,code):
    # poison the query
    # The suffix should be added to the last token of the query (not . or ,)
    new_query = []
    poisoned = False
    for token in query[-1::-1]:
        if poisoned or token in ['.', ',', '(', ')', '[', ']', '{', '}', ':', ';']:
            new_query.insert(0, token)
        else:
            new_query.insert(0, token + "_" + targets[0])
            poisoned = True
    # poison the code
    # add the suffix at the end of function name and the variable name
    try:
        tree = ast.parse(code)

        # create CodeVisitor
        visitor = CodeVisitor(targets)
        new_tree = visitor.visit(tree)
        new_code = ast.unparse(new_tree)
    except Exception:
        return query, code
    return new_query, new_code

if __name__ == "__main__":
    # words = ["apple", "apply", "banana", "band", "cat", "bat", "rat", "mat"]
    # labels = cluster_words_unsupervised(words, eps=2, min_samples=2)
    #
    # 
    # clusters = {}
    # for word, label in zip(words, labels):
    #     clusters.setdefault(label, []).append(word)
    #
    # print(clusters)
    # 示例代码
    code = """
def my_function(x):
    y = x + 1
    z = y * 2
    return z
"""

    
    tree = ast.parse(code)

    
    words = ["apple", "apply", "banana", "band", "cat", "bat", "rat", "mat"]
    visitor = CodeVisitor(words)
    new_tree = visitor.visit(tree)

    
    new_code = ast.unparse(new_tree)
    print(new_code)