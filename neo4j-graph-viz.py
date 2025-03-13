#!/usr/bin/env python3
"""
Neo4j Graph Visualization Script for Government Data Pipeline.
Extracts and visualizes knowledge graph structures.
"""

import os
import sys
import argparse
import logging
import json
import random
from dotenv import load_dotenv
from tabulate import tabulate
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Try to import Neo4j driver
try:
    from neo4j import GraphDatabase
except ImportError:
    logger.error("Could not import Neo4j driver. Make sure it's installed (pip install neo4j).")
    sys.exit(1)

class Neo4jGraphVisualizer:
    """Class to visualize Neo4j graphs."""
    
    def __init__(self, uri, username, password):
        """Initialize Neo4j connection."""
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Verify connection
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def run_query(self, query, params=None):
        """Run a Cypher query and return results."""
        if not params:
            params = {}
            
        with self.driver.session() as session:
            result = session.run(query, params)
            return list(result)
    
    def get_graph_summary(self):
        """Get a summary of the knowledge graph."""
        # Get node counts by type
        node_count_query = """
        MATCH (n)
        RETURN labels(n) AS label, count(*) AS count
        """
        
        # Get relationship counts by type
        rel_count_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS count
        """
        
        # Get top entities by connections
        top_entities_query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) AS rel_count
        RETURN e.canonical_name AS name, e.type AS type, rel_count
        ORDER BY rel_count DESC
        LIMIT 10
        """
        
        node_counts = self.run_query(node_count_query)
        rel_counts = self.run_query(rel_count_query)
        top_entities = self.run_query(top_entities_query)
        
        # Process node counts
        node_count_dict = {}
        for record in node_counts:
            labels = record["label"]
            count = record["count"]
            label_str = ", ".join(labels) if labels else "No Label"
            node_count_dict[label_str] = count
        
        # Process relationship counts
        rel_count_dict = {}
        for record in rel_counts:
            rel_type = record["type"]
            count = record["count"]
            rel_count_dict[rel_type] = count
        
        # Process top entities
        top_entities_list = []
        for record in top_entities:
            top_entities_list.append({
                "name": record["name"],
                "type": record["type"],
                "connections": record["rel_count"]
            })
        
        return {
            "node_counts": node_count_dict,
            "relationship_counts": rel_count_dict,
            "top_entities": top_entities_list
        }
    
    def extract_subgraph(self, entity_name=None, limit=50):
        """
        Extract a subgraph from Neo4j.
        If entity_name is provided, extract the neighborhood of that entity.
        Otherwise, extract a sample subgraph.
        """
        if entity_name:
            # Extract neighborhood of the specified entity
            query = """
            MATCH path = (e:Entity {canonical_name: $name})-[r*0..2]-(other)
            RETURN path
            LIMIT $limit
            """
            params = {"name": entity_name, "limit": limit}
        else:
            # Extract a sample subgraph
            query = """
            MATCH path = (e:Entity)-[r]-(other)
            RETURN path
            LIMIT $limit
            """
            params = {"limit": limit}
        
        # Run the query
        results = self.run_query(query, params)
        
        # Build a networkx graph from the results
        G = nx.DiGraph()
        
        for record in results:
            path = record["path"]
            
            # Add all nodes and relationships from the path
            for node in path.nodes:
                # Extract useful properties
                props = dict(node.items())
                labels = list(node.labels)
                
                # Create a label for the node
                if "canonical_name" in props:
                    node_label = props["canonical_name"]
                elif "title" in props:
                    node_label = props["title"][:30] + "..." if len(props["title"]) > 30 else props["title"]
                else:
                    node_label = f"Node_{node.id}"
                
                # Add the node with its properties
                G.add_node(
                    node.id,
                    label=node_label,
                    type=labels[0] if labels else "Unknown",
                    properties=props
                )
            
            # Add all relationships
            for rel in path.relationships:
                # Extract the relationship type and properties
                rel_type = rel.type
                props = dict(rel.items())
                
                # Add the relationship
                G.add_edge(
                    rel.start_node.element_id,
                    rel.end_node.element_id,
                    type=rel_type,
                    properties=props
                )
        
        return G
    
    def export_graphml(self, graph, filename="knowledge_graph.graphml"):
        """Export the graph to GraphML format."""
        # Convert node and edge attributes to strings for GraphML compatibility
        for node in graph.nodes:
            for k, v in graph.nodes[node].items():
                if isinstance(v, dict) or isinstance(v, list):
                    graph.nodes[node][k] = json.dumps(v)
        
        for u, v, data in graph.edges(data=True):
            for k, v in data.items():
                if isinstance(v, dict) or isinstance(v, list):
                    data[k] = json.dumps(v)
        
        # Export the graph
        nx.write_graphml(graph, filename)
        logger.info(f"Graph exported to {filename}")
    
    def print_ascii_graph(self, graph, max_nodes=20):
        """Print an ASCII representation of the graph."""
        # Limit the number of nodes to display
        if len(graph.nodes) > max_nodes:
            logger.info(f"Graph has {len(graph.nodes)} nodes, showing only {max_nodes} nodes")
            
            # Get the most connected nodes
            nodes_by_connections = sorted(
                graph.nodes, 
                key=lambda n: graph.degree(n), 
                reverse=True
            )[:max_nodes]
            
            # Create a subgraph
            subgraph = graph.subgraph(nodes_by_connections)
        else:
            subgraph = graph
        
        # Print nodes
        print("\nNodes:")
        for node in subgraph.nodes:
            node_data = subgraph.nodes[node]
            label = node_data.get("label", f"Node_{node}")
            node_type = node_data.get("type", "Unknown")
            print(f"  - {label} ({node_type})")
        
        # Print edges
        print("\nRelationships:")
        for u, v, data in subgraph.edges(data=True):
            u_label = subgraph.nodes[u].get("label", f"Node_{u}")
            v_label = subgraph.nodes[v].get("label", f"Node_{v}")
            rel_type = data.get("type", "Unknown")
            print(f"  - {u_label} --[{rel_type}]--> {v_label}")
    
    def visualize_graph(self, graph, filename="knowledge_graph.png", layout="spring"):
        """Visualize the graph using matplotlib."""
        # Create a new figure
        plt.figure(figsize=(12, 10))
        
        # Choose a layout
        if layout == "spring":
            pos = nx.spring_layout(graph, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "shell":
            pos = nx.shell_layout(graph)
        else:
            pos = nx.kamada_kawai_layout(graph)
        
        # Extract node types
        node_types = set(nx.get_node_attributes(graph, "type").values())
        
        # Generate colors for node types
        color_map = {}
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, node_type in enumerate(node_types):
            color_map[node_type] = colors[i % len(colors)]
        
        # Extract edge types
        edge_types = set(nx.get_edge_attributes(graph, "type").values())
        
        # Generate colors for edge types
        edge_color_map = {}
        for i, edge_type in enumerate(edge_types):
            edge_color_map[edge_type] = colors[(i + len(node_types)) % len(colors)]
        
        # Draw nodes by type
        for node_type in node_types:
            node_list = [n for n, data in graph.nodes(data=True) if data.get("type") == node_type]
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=node_list,
                node_color=color_map[node_type],
                node_size=500,
                alpha=0.8,
                label=node_type
            )
        
        # Draw edges by type
        for edge_type in edge_types:
            edge_list = [(u, v) for u, v, data in graph.edges(data=True) if data.get("type") == edge_type]
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=edge_list,
                edge_color=edge_color_map[edge_type],
                width=1.5,
                alpha=0.7,
                label=edge_type
            )
        
        # Draw node labels
        labels = nx.get_node_attributes(graph, "label")
        nx.draw_networkx_labels(
            graph, pos,
            labels=labels,
            font_size=8,
            font_weight="bold"
        )
        
        # Add legend for node types
        plt.legend(title="Node and Edge Types")
        
        # Save the figure
        plt.title("Knowledge Graph Visualization")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Graph visualization saved to {filename}")
        
        return filename

def get_connection_details():
    """Get Neo4j connection details from environment variables or user input."""
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri:
        uri = input("Enter Neo4j URI (e.g., neo4j+s://xxx.databases.neo4j.io): ")
    
    if not username:
        username = input("Enter Neo4j username: ")
    
    if not password:
        import getpass
        password = getpass.getpass("Enter Neo4j password: ")
    
    return uri, username, password

def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(description='Neo4j Graph Visualization Script')
    parser.add_argument('--summary', action='store_true', help='Print a summary of the knowledge graph')
    parser.add_argument('--entity', metavar='NAME', help='Extract and visualize the neighborhood of a specific entity')
    parser.add_argument('--sample', action='store_true', help='Extract and visualize a sample subgraph')
    parser.add_argument('--limit', type=int, default=50, help='Maximum number of relationships to extract')
    parser.add_argument('--output', metavar='FILE', help='Filename for the output visualization (default: auto-generated)')
    parser.add_argument('--format', choices=['png', 'graphml', 'ascii'], default='png',
                       help='Output format (default: png)')
    parser.add_argument('--layout', choices=['spring', 'circular', 'shell', 'kamada_kawai'], default='spring',
                       help='Graph layout algorithm (default: spring)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    # Get Neo4j connection details
    uri, username, password = get_connection_details()
    
    try:
        # Initialize visualizer
        visualizer = Neo4jGraphVisualizer(uri, username, password)
        
        # Process commands
        if args.summary:
            summary = visualizer.get_graph_summary()
            
            print("\n===== Knowledge Graph Summary =====")
            
            print("\nNode Counts:")
            node_data = [[label, count] for label, count in summary["node_counts"].items()]
            print(tabulate(node_data, headers=["Node Type", "Count"], tablefmt="grid"))
            
            print("\nRelationship Counts:")
            rel_data = [[rel_type, count] for rel_type, count in summary["relationship_counts"].items()]
            print(tabulate(rel_data, headers=["Relationship Type", "Count"], tablefmt="grid"))
            
            print("\nTop Entities by Connections:")
            entity_data = [
                [entity["name"], entity["type"], entity["connections"]] 
                for entity in summary["top_entities"]
            ]
            print(tabulate(entity_data, headers=["Entity", "Type", "Connections"], tablefmt="grid"))
        
        elif args.entity or args.sample:
            # Extract the subgraph
            if args.entity:
                graph = visualizer.extract_subgraph(args.entity, args.limit)
                source = f"entity '{args.entity}'"
            else:
                graph = visualizer.extract_subgraph(limit=args.limit)
                source = "random sample"
            
            print(f"\nExtracted subgraph from {source} with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            
            # Generate output filename if not specified
            if not args.output:
                timestamp = random.randint(1000, 9999)
                if args.entity:
                    base_name = f"graph_{args.entity.replace(' ', '_')}_{timestamp}"
                else:
                    base_name = f"graph_sample_{timestamp}"
                
                if args.format == 'png':
                    args.output = f"{base_name}.png"
                elif args.format == 'graphml':
                    args.output = f"{base_name}.graphml"
                else:
                    args.output = None
            
            # Generate the visualization
            if args.format == 'png':
                filename = visualizer.visualize_graph(graph, args.output, args.layout)
                print(f"Visualization saved to {filename}")
            elif args.format == 'graphml':
                visualizer.export_graphml(graph, args.output)
                print(f"GraphML file saved to {args.output}")
            else:  # ascii
                visualizer.print_ascii_graph(graph)
        
        # Close connection
        visualizer.close()
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()