## [URI-Graph Dataset](https://arxiv.org/pdf/2205.14005)
Please first [download](https://drive.google.com/drive/folders/1DvjFUa9gGqalrFAYFzbB2onGXU2K4LNu) the data, then put it in folder ```data/```

The URI-Graph dataset is a heterogeneous network that captures the relationships among three main entities involved in recipe recommendation:  
	•	**Users (7,958 nodes)**: Represent individuals who rate or interact with recipes.  
	•	**Recipes (68,794 nodes)**: Each recipe is associated with relevant attributes such as instructions and ingredients.  
	•	**Ingredients (8,847 nodes)**: Matched to USDA’s nutritional database to capture nutritional information.  

The dataset contains four types of edges:  
	1.	**User–Recipe (135,353 edges)**: Connects users to recipes they rated; the rating score serves as the edge weight.  
	2.	**Recipe–Recipe (647,146 edges)**: Links recipes by their similarity (via FoodKG and Reciptor), using the similarity score as the edge weight.  
	3.	**Recipe–Ingredient (463,485 edges)**: Indicates which ingredients appear in a recipe, with each ingredient’s usage weight on the edge.  
	4.	**Ingredient–Ingredient (146,188 edges)**: Reflects how frequently two ingredients co-occur based on Normalized Pointwise Mutual Information (NPMI).  

Through these nodes and edges, URI-Graph encapsulates users’ preferences, recipe similarities, ingredient usage and ingredient co-occurrences, forming a rich heterogeneous network for advanced recipe recommendation and related tasks.
