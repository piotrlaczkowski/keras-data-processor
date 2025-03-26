#!/usr/bin/env python
"""
Script to extract docstrings from the codebase and generate Markdown documentation.
This allows for automatic documentation generation directly from the code.
"""

import os
import re
import inspect
import sys
from pathlib import Path

# Add the project root to the Python path to allow module imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import required modules


def docstring_to_markdown(docstring):
    """
    Convert a Python docstring to Markdown format.

    Args:
        docstring: Raw docstring from the Python object.

    Returns:
        str: Markdown-formatted documentation.
    """
    if not docstring:
        return ""

    # Remove leading spaces/indentation from each line
    lines = docstring.split("\n")
    if len(lines) > 1:
        # Find minimum indentation (excluding empty lines)
        min_indent = min(
            (len(line) - len(line.lstrip(" ")) for line in lines[1:] if line.strip()),
            default=0,
        )
        # Remove this indentation from all lines
        docstring = "\n".join(
            [lines[0]]
            + [line[min_indent:] if line.strip() else line for line in lines[1:]]
        )

    # Replace reST/Google-style formatting with Markdown equivalents
    markdown = docstring

    # Convert Args: section to Markdown
    markdown = re.sub(
        r"Args:\s*\n((?:\s+\w+(?:\s*\([^)]*\))?\s*:.*\n(?:\s+[^\n]*\n)*)*)",
        r"### Parameters\n\n\1",
        markdown,
    )

    # Convert Returns: section to Markdown
    markdown = re.sub(
        r"Returns:\s*\n(\s+[^\n]*\n(?:\s+[^\n]*\n)*)", r"### Returns\n\n\1", markdown
    )

    # Convert Raises: section to Markdown
    markdown = re.sub(
        r"Raises:\s*\n(\s+[^\n]*\n(?:\s+[^\n]*\n)*)", r"### Raises\n\n\1", markdown
    )

    # Convert Examples: section to Markdown
    markdown = re.sub(
        r"Examples?:\s*\n(\s+[^\n]*\n(?:\s+[^\n]*\n)*)", r"### Examples\n\n\1", markdown
    )

    # Convert Note: section to Markdown
    markdown = re.sub(
        r"Note:\s*\n(\s+[^\n]*\n(?:\s+[^\n]*\n)*)", r"### Notes\n\n\1", markdown
    )

    # Convert parameter descriptions to bullet points
    def param_replace(match):
        param_name = match.group(1)
        param_desc = match.group(2).strip()
        return f"- **{param_name}**: {param_desc}\n"

    markdown = re.sub(
        r"\s+(\w+(?:\s*\([^)]*\))?)\s*:(.*?)(?=\n\s+\w+\s*:|$)",
        param_replace,
        markdown,
        flags=re.DOTALL,
    )

    # Remove trailing spaces
    markdown = "\n".join(line.rstrip() for line in markdown.split("\n"))

    return markdown


def extract_class_docs(cls):
    """
    Extract documentation from a class, including its methods.

    Args:
        cls: The class to document

    Returns:
        str: Markdown documentation for the class and its methods
    """
    docs = f"# {cls.__name__}\n\n"

    # Add class docstring
    if cls.__doc__:
        docs += f"{docstring_to_markdown(cls.__doc__)}\n\n"

    # Document public methods
    methods = []
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        # Skip private methods
        if name.startswith("_") and name != "__init__":
            continue

        methods.append((name, method))

    # Sort methods alphabetically, but keep __init__ first
    methods.sort(key=lambda x: (0 if x[0] == "__init__" else 1, x[0]))

    # Add method documentation
    for name, method in methods:
        if method.__doc__:
            method_name = "Constructor" if name == "__init__" else name
            docs += f"## {method_name}\n\n"
            docs += f"```python\n{name}{inspect.signature(method)}\n```\n\n"
            docs += f"{docstring_to_markdown(method.__doc__)}\n\n"
            docs += "---\n\n"

    return docs


def extract_module_docs(module, output_dir):
    """
    Extract documentation from all classes in a module.

    Args:
        module: The module to document
        output_dir: Directory to save the documentation
    """
    module_name = module.__name__.split(".")[-1]

    # Create output directory if it doesn't exist
    (Path(output_dir) / "api").mkdir(exist_ok=True, parents=True)

    # Find all classes in the module
    for name, obj in inspect.getmembers(module, predicate=inspect.isclass):
        # Skip imported classes
        if obj.__module__ != module.__name__:
            continue

        # Generate documentation
        docs = extract_class_docs(obj)

        # Write to file
        output_file = Path(output_dir) / "api" / f"{module_name}_{name}.md"
        with open(output_file, "w") as f:
            f.write(docs)

        print(f"Generated docs for {module.__name__}.{name} at {output_file}")


def generate_module_index(modules, output_dir):
    """
    Generate an index file listing all documented modules and classes.

    Args:
        modules: List of modules that have been documented
        output_dir: Directory where documentation is saved
    """
    output_file = Path(output_dir) / "api_index.md"

    with open(output_file, "w") as f:
        f.write("# API Reference\n\n")
        f.write(
            "This section provides detailed API documentation extracted directly from the codebase.\n\n"
        )

        for module in modules:
            module_name = module.__name__
            f.write(f"## {module_name}\n\n")

            # Find all documented classes in this module
            doc_dir = Path(output_dir) / "api"
            module_short_name = module_name.split(".")[-1]

            # List documented classes
            for doc_file in sorted(doc_dir.glob(f"{module_short_name}_*.md")):
                class_name = doc_file.stem.split("_", 1)[1]
                relative_path = os.path.relpath(doc_file, Path(output_dir))
                f.write(f"- [{class_name}]({relative_path})\n")

            f.write("\n")

    print(f"Generated API index at {output_file}")


def extract_docstrings():
    """
    Main function to extract docstrings from the codebase and generate documentation.
    """
    # Define output directory
    output_dir = "docs/generated"

    # List of modules to document
    modules = [
        sys.modules["kdp.processor"],
        sys.modules["kdp.dynamic_pipeline"],
        sys.modules["kdp.features"],
        sys.modules["kdp.layers.distribution_aware_encoder_layer"],
        sys.modules["kdp.layers.distribution_transform_layer"],
        sys.modules["kdp.layers.global_numerical_embedding_layer"],
        sys.modules["kdp.layers.numerical_embedding_layer"],
        sys.modules["kdp.layers.tabular_attention_layer"],
        sys.modules["kdp.layers.multi_resolution_tabular_attention_layer"],
    ]

    # Process each module
    for module in modules:
        extract_module_docs(module, output_dir)

    # Generate index
    generate_module_index(modules, output_dir)

    print("Documentation generation complete!")


if __name__ == "__main__":
    extract_docstrings()
