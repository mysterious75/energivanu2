import re

with open('D:/hacker/energivanu/Energivanu2/magazine/build_magazine.py', 'r') as f:
    content = f.read()

# Remove the remaining add_font calls
content = re.sub(r"self\.add_font.*?\n", "", content)

# Replace all uses of 'LibSans' with 'helvetica'
content = content.replace("'LibSans'", "'helvetica'")

with open('D:/hacker/energivanu/Energivanu2/magazine/build_magazine.py', 'w') as f:
    f.write(content)

print("Font paths fixed and changed to helvetica.")
