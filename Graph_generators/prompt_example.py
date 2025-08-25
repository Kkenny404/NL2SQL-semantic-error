# Generate two thesis-ready figures from the user's CoT prompt using matplotlib.
# 1) A "Prompt Template Card" that neatly lays out the sections.
# 2) A simple flowchart showing the analysis pipeline ending in a Yes/No decision.
#
# Both figures will be exported as high-resolution PNG and vector PDF for print quality.

import textwrap
import matplotlib.pyplot as plt

# ---- User prompt content (copied from the message) ----
cot_prompt = """You are a database expert. Analyze whether the SQL query correctly answers the natural language question. 

Question: {question}

Database Schema: {schema}

SQL Query: {sql}

Analyze this step by step in your mind (do not output the steps, just the final answer):

1. Question Understanding: What exactly is the question asking for?
   - What information should be retrieved?
   - Are there any specific conditions or constraints?

2. SQL Logic Analysis: Break down what the SQL actually does:
   - Which tables and columns are selected?
   - What are the JOIN conditions?
   - What filters are applied in WHERE clause?
   - Are aggregations/sorting correctly used?

3. Semantic Correctness Check: 
   - Do the selected columns semantically match what the question requests?
   - Are the table relationships and JOIN logic semantically appropriate?
   - Do the WHERE conditions semantically align with the question's intent?
   - Does the overall query logic semantically represent the question's meaning?

4. Final Verification: Does the SQL query produce the exact information requested in the question?

Final Answer: [Yes or No]. Answer one word only, without any explanation or additional text.
"""

# --- Helper: wrapped text inside a box ---
def add_boxed_text(ax, x, y, w, h, title, body, fontsize=10):
    # Rectangle
    rect = plt.Rectangle((x, y), w, h, fill=False)
    ax.add_patch(rect)
    # Title
    ax.text(x + 0.02*w, y + h - 0.15*h, title, fontsize=12, fontweight='bold', va='top')
    # Body
    wrap = textwrap.fill(body, width=70)
    ax.text(x + 0.02*w, y + h - 0.20*h, wrap, fontsize=fontsize, va='top')

# ========== Figure 1: Prompt Template Card ==========
fig1 = plt.figure(figsize=(8.27, 11.69))  # A4 ratio in inches (portrait)
ax1 = fig1.add_axes([0.06, 0.05, 0.88, 0.90])
ax1.axis('off')

# Title
ax1.text(0.0, 0.98, "Chain-of-Thought Prompt Template (SQL Semantic Validation)",
         fontsize=16, fontweight='bold', va='top')

# Intro line
intro = ("Role: Database expert; Task: judge whether the SQL query answers the natural language question.")
ax1.text(0.0, 0.94, intro, fontsize=11, va='top')

# Sections
sections = [
    ("Inputs", "Question: {question}\nDatabase Schema: {schema}\nSQL Query: {sql}"),
    ("Instruction", "Analyze this step by step in your mind (do not output the steps, just the final answer)."),
    ("Step 1 — Question Understanding",
     "What exactly is the question asking for?\n- What information should be retrieved?\n- Any specific conditions or constraints?"),
    ("Step 2 — SQL Logic Analysis",
     "Break down what the SQL actually does:\n- Which tables and columns are selected?\n- What are the JOIN conditions?\n- What filters are applied in WHERE?\n- Are aggregations/sorting correctly used?"),
    ("Step 3 — Semantic Correctness Check",
     "- Do selected columns match the question request?\n- Are table relationships / JOIN logic appropriate?\n- Do WHERE conditions align with intent?\n- Does the overall logic represent the question meaning?"),
    ("Step 4 — Final Verification",
     "Does the SQL produce exactly what the question requests?"),
    ("Output", "Final Answer: [Yes or No]. Output one word only, no explanations.")
]

# Layout grid
y = 0.88
box_h = 0.10
for i, (title, body) in enumerate(sections):
    add_boxed_text(ax1, x=0.0, y=y - box_h, w=1.0, h=box_h, title=title, body=body)
    y -= (box_h + 0.02)

# Save
fig1.savefig("/mnt/data/cot_prompt_template_card.png", dpi=300, bbox_inches="tight")
fig1.savefig("/mnt/data/cot_prompt_template_card.pdf", bbox_inches="tight")
plt.close(fig1)

# ========== Figure 2: Prompt Flowchart ==========
fig2 = plt.figure(figsize=(11.69, 8.27))  # A4 ratio in inches (landscape)
ax2 = fig2.add_axes([0.03, 0.03, 0.94, 0.94])
ax2.axis('off')

ax2.text(0.5, 0.98, "Analysis Flow for CoT Prompt (SQL Semantic Validation)",
         fontsize=16, fontweight='bold', ha='center', va='top')

# Node positions (x_center, y_center)
nodes = {
    "Inputs": (0.20, 0.80),
    "Step1": (0.40, 0.80),
    "Step2": (0.60, 0.80),
    "Step3": (0.80, 0.80),
    "Verify": (0.50, 0.50),
    "Output": (0.50, 0.20)
}

def draw_node(ax, label, center, width=0.28, height=0.16, fontsize=10):
    x = center[0] - width/2
    y = center[1] - height/2
    rect = plt.Rectangle((x, y), width, height, fill=False)
    ax.add_patch(rect)
    ax.text(center[0], center[1], textwrap.fill(label, width=42),
            ha='center', va='center', fontsize=fontsize)

def draw_arrow(ax, start, end):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", shrinkA=10, shrinkB=10)
    )

# Node labels
labels = {
    "Inputs": "Inputs\n• Question\n• Database Schema\n• SQL Query",
    "Step1": "Step 1: Question Understanding\nWhat is asked? Info needed? Constraints?",
    "Step2": "Step 2: SQL Logic Analysis\nTables/columns, JOINs, WHERE filters, aggregation/sorting",
    "Step3": "Step 3: Semantic Correctness Check\nColumns match intent? JOINs appropriate? WHERE aligned? Overall meaning correct?",
    "Verify": "Step 4: Final Verification\nDoes the SQL produce exactly what the question requests?",
    "Output": "Output\nFinal Answer: Yes / No (one word)"
}

# Draw nodes
for key in nodes:
    draw_node(ax2, labels[key], nodes[key])

# Draw arrows
draw_arrow(ax2, (nodes["Inputs"][0]+0.14, nodes["Inputs"][1]), (nodes["Step1"][0]-0.14, nodes["Step1"][1]))
draw_arrow(ax2, (nodes["Step1"][0]+0.14, nodes["Step1"][1]), (nodes["Step2"][0]-0.14, nodes["Step2"][1]))
draw_arrow(ax2, (nodes["Step2"][0]+0.14, nodes["Step2"][1]), (nodes["Step3"][0]-0.14, nodes["Step3"][1]))
draw_arrow(ax2, (0.50, 0.72), nodes["Verify"])
draw_arrow(ax2, nodes["Verify"], nodes["Output"])

# Save
fig2.savefig("/mnt/data/cot_prompt_flowchart.png", dpi=300, bbox_inches="tight")
fig2.savefig("/mnt/data/cot_prompt_flowchart.pdf", bbox_inches="tight")
plt.close(fig2)

print("Created files:\n- /mnt/data/cot_prompt_template_card.png\n- /mnt/data/cot_prompt_template_card.pdf\n- /mnt/data/cot_prompt_flowchart.png\n- /mnt/data/cot_prompt_flowchart.pdf")
