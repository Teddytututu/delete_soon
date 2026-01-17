import json
import re
import pypandoc

# A sample Markdown string

markdown_text = """
# My Document


Some **bold** text here, and some *italic* text there.

- Bullet point 1
- Bullet point 2
"""


def markdown_to_latex(markdown_text):
    # Convert Markdown string to LaTeX

    latex_text = pypandoc.convert_text(markdown_text, to='latex', format='md')
    return latex_text


def markdown_to_json_method(markdown_text):
    #  0

    root = {"method_class": "root", "children": []}
    stack = [{"node": root, "level": 0}]
    
    lines = markdown_text.strip().split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        if not line:
            continue
        

        if line.startswith('#'):

            match = re.match(r'^(#+)\s*(.*?)$', line)

            if not match:
                continue
            hashes, method_class = match.groups()
            current_level = len(hashes)
            

            new_node = {"method_class": method_class, "children": [], "description": ""}
            

            while stack and stack[-1]["level"] >= current_level:
                stack.pop()
            
            #  new_node  root 

            if stack:
                parent = stack[-1]["node"]
            else:
                parent = root
            parent["children"].append(new_node)
            

            stack.append({"node": new_node, "level": current_level})
            

            description_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('#') and not lines[i].strip().startswith('-'):

                description_lines.append(lines[i].strip())
                i += 1
            
            if description_lines:
                new_node["description"] = " ".join(description_lines)

            # NOTE:  i ""//

            #  lines[i]


        

        elif line.startswith('-'):
            item = {}
            if ': ' in line:
                method, description = line[1:].strip().split(': ', 1)
                description = description
                item = {"method": method.strip(), "description": description.strip()}
            else:
                item = {"method": line[1:].strip(), "description": ""}
            
            #  root

            if stack:
                current_node = stack[-1]["node"]
                current_node.setdefault("children", []).append(item)
            else:
                root.setdefault("children", []).append(item)
    

    return root["children"]


if __name__ == "__main__":
    with open("../data/actor_data/docs/method_en_v1.md", "r", encoding="utf-8") as f:
        markdown_text = f.read()

    result = markdown_to_json_method(markdown_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))

