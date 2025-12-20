import collections

def bfs(graph, start_node):
    """
    Effectue une recherche en largeur d'abord (BFS) sur un graphe.

    Args:
        graph (dict): Le graphe représenté comme une liste d'adjacence.
        start_node: Le nœud de départ de la traversée.
    """
    
    # Un ensemble (set) pour garder une trace des nœuds visités afin d'éviter les cycles et le travail redondant.
    visited = set()
    
    # Une file d'attente (FIFO - Premier Entré, Premier Sorti) pour gérer les nœuds à visiter.
    # Nous utilisons collections.deque pour une implémentation efficace de la file d'attente.
    queue = collections.deque([start_node])
    
    # Ajouter le nœud de départ aux nœuds visités immédiatement.
    visited.add(start_node)
    
    print("Démarrage de la Recherche en Largeur d'Abord (BFS):")
    
    while queue:
        # Défiler un nœud du début de la file d'attente
        node = queue.popleft()
        print(f"Visite de : {node}")

        # Obtenir tous les voisins du nœud actuel.
        # Utiliser graph.get(node, []) pour gérer les nœuds sans voisins.
        for neighbor in graph.get(node, []):
            # Si le voisin n'a pas encore été visité
            if neighbor not in visited:
                # Le marquer comme visité
                visited.add(neighbor)
                # L'enfiler pour être visité plus tard
                queue.append(neighbor)
    
    print("BFS terminée.\n")


def dfs_recursive(graph, node, visited=None):
    """
    Effectue une recherche en profondeur d'abord (DFS) sur un graphe en utilisant la récursivité.

    Args:
        graph (dict): Le graphe représenté comme une liste d'adjacence.
        node: Le nœud actuel visité.
        visited (set): Un ensemble de nœuds déjà visités.
    """
    
    # Initialiser l'ensemble des nœuds visités lors du premier appel
    if visited is None:
        visited = set()
        print("Démarrage de la Recherche en Profondeur d'Abord (DFS) - Récursive:")

    # Marquer le nœud actuel comme visité
    visited.add(node)
    print(f"Visite de : {node}")

    # Appel récursif pour tous les voisins non visités
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)


def dfs_iterative(graph, start_node):
    """
    Effectue une recherche en profondeur d'abord (DFS) sur un graphe en utilisant une pile explicite (itératif).

    Args:
        graph (dict): Le graphe représenté comme une liste d'adjacence.
        start_node: Le nœud de départ de la traversée.
    """
    
    # Un ensemble pour garder une trace des nœuds visités.
    visited = set()
    
    # Une pile (LIFO - Dernier Entré, Premier Sorti) pour gérer les nœuds à visiter.
    stack = [start_node]
    
    print("Démarrage de la Recherche en Profondeur d'Abord (DFS) - Itérative:")
    
    while stack:
        # Dépiler un nœud du sommet de la pile
        node = stack.pop()
        
        # Important : Vérifier si le nœud est visité *après* l'avoir dépilé.
        # Ceci gère les cas où un nœud est ajouté à la pile plusieurs
        # fois avant d'être visité (par exemple, comme voisin de plusieurs nœuds).
        if node not in visited:
            print(f"Visite de : {node}")
            # Le marquer comme visité
            visited.add(node)
            
            # Ajouter tous les voisins non visités à la pile.
            # Nous les ajoutons en ordre inverse afin que le "premier" voisin
            # (dans la liste d'adjacence) soit traité en premier,
            # ce qui correspond souvent au comportement de la version récursive.
            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    print("DFS Itérative terminée.\n")


# --- Exemple d'Utilisation ---

if __name__ == "__main__":
    # Définir un exemple de graphe comme liste d'adjacence
    # 'A': ['B', 'C'] signifie que le nœud 'A' a des arêtes vers 'B' et 'C'
    example_graph = {
        'A': ['B', 'C', 'D'],
        'B': ['A', 'E'],
        'C': ['A', 'F', 'G'],
        'D': ['A'],
        'E': ['B'],
        'F': ['C'],
        'G': ['C', 'H'],
        'H': ['G']
    }
    
    # --- Exécuter BFS ---
    # BFS visitera par niveaux :
    # Niveau 0: A
    # Niveau 1: B, C, D
    # Niveau 2: E, F, G
    # Niveau 3: H
    bfs(example_graph, 'A')

    # --- Exécuter DFS (Récursive) ---
    # DFS ira d'abord en profondeur, par exemple : A -> B -> E -> (retour arrière) -> C -> F ...
    dfs_recursive(example_graph, 'A')
    print("DFS Récursive terminée.\n") # Ajouter un saut de ligne pour la clarté

    # --- Exécuter DFS (Itérative) ---
    # Produira le même ensemble de nœuds visités, souvent dans un
    # ordre similaire (bien que pas toujours identique) à la version récursive.
    dfs_iterative(example_graph, 'A')