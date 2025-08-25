import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import textwrap

def load_taxonomy_data(json_file):
    """Load taxonomy data from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_simple_table(data, output_file='nl2sql_taxonomy_table.png'):
    """Create a simple, clean table layout."""
    
    colors = [
        '#dc2626',  # Red
        '#ea580c',  # Orange-red  
        '#d97706',  # Orange
        '#16a34a',  # Green
        '#0891b2',  # Cyan
        '#7c3aed',  # Purple
        '#be185d',  # Pink
        '#059669',  # Emerald
        '#6366f1'   # Indigo
    ]
    
    # Calculate row heights based on sub-error counts
    sub_error_counts = [len(cat['sub_errors']) for cat in data]
    base_row_height = 0.8
    height_per_sub_error = 0.3
    row_heights = [base_row_height + count * height_per_sub_error for count in sub_error_counts]
    total_height = sum(row_heights) + 2
    
    fig, ax = plt.subplots(figsize=(18, total_height))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, total_height)
    
    # Title
    ax.text(9, total_height - 0.5, 'A Taxonomy of NL2SQL Translation Semantic Errors', 
            fontsize=20, fontweight='bold', ha='center', va='center')
    
    # Column headers
    header_y = total_height - 1.2
    ax.text(2.5, header_y, 'Error Category', fontsize=14, fontweight='bold', ha='center')
    ax.text(13, header_y, 'Sub-error Types', fontsize=14, fontweight='bold', ha='center')
    
    # Header line
    ax.axhline(y=header_y - 0.2, color='black', linewidth=2)
    
    current_y = header_y - 0.4
    
    for i, category in enumerate(data):
        color = colors[i % len(colors)]
        row_height = row_heights[i]
        
        # Category column (left)
        cat_rect = Rectangle((0.5, current_y - row_height), 4.5, row_height,
                           facecolor=color, alpha=0.1, edgecolor=color, linewidth=2)
        ax.add_patch(cat_rect)
        
        # Category title
        ax.text(0.7, current_y - 0.3, category['category'], 
                fontsize=13, fontweight='bold', color=color, va='top', 
                wrap=True)
        
        # Category description
        desc = category['description'][:150] + "..." if len(category['description']) > 150 else category['description']
        wrapped_desc = '\n'.join(textwrap.wrap(desc, width=40))
        ax.text(0.7, current_y - 0.6, wrapped_desc, 
                fontsize=9, color='#555555', va='top')
        
        # Sub-errors column (right)
        sub_rect = Rectangle((5.5, current_y - row_height), 12, row_height,
                           facecolor='white', edgecolor=color, linewidth=1.5, alpha=0.9)
        ax.add_patch(sub_rect)
        
        # List sub-errors vertically
        sub_y_start = current_y - 0.2
        sub_line_height = 0.35
        
        for j, sub_error in enumerate(category['sub_errors']):
            sub_y = sub_y_start - j * sub_line_height
            
            # Sub-error name with bullet point
            ax.text(5.8, sub_y, f"• {sub_error['error_type']}", 
                    fontsize=11, fontweight='bold', color=color, va='top')
            
            # Sub-error description
            desc = sub_error['description'][:120] + "..." if len(sub_error['description']) > 120 else sub_error['description']
            wrapped_desc = '\n'.join(textwrap.wrap(desc, width=80))
            ax.text(6.0, sub_y - 0.15, wrapped_desc, 
                    fontsize=9, color='#333333', va='top')
        
        current_y -= row_height + 0.1  # Small gap between rows
        
        # Row separator
        if i < len(data) - 1:
            ax.axhline(y=current_y + 0.05, color='lightgray', linewidth=1, alpha=0.5)
    
    # Table border
    ax.add_patch(Rectangle((0.5, 0.2), 17, total_height - 1.5, 
                          fill=False, edgecolor='black', linewidth=2))
    
    # Vertical separator between columns
    ax.axvline(x=5.5, ymin=0.2/total_height, ymax=(total_height-1.5)/total_height, 
               color='black', linewidth=2)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_compact_table(data, output_file='nl2sql_taxonomy_compact_table.png'):
    """Create a more compact table version."""
    
    colors = ['#dc2626', '#ea580c', '#d97706', '#16a34a', '#0891b2', 
              '#7c3aed', '#be185d', '#059669', '#6366f1']
    
    # Fixed row height for compact version
    row_height = 1.0
    total_height = len(data) * row_height + 2.5
    
    fig, ax = plt.subplots(figsize=(16, total_height))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, total_height)
    
    # Title
    ax.text(8, total_height - 0.5, 'NL2SQL Translation Semantic Error Taxonomy', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Table headers
    header_y = total_height - 1.2
    header_rect = Rectangle((0.5, header_y - 0.3), 15, 0.6,
                          facecolor='#f0f0f0', edgecolor='black', linewidth=2)
    ax.add_patch(header_rect)
    
    ax.text(3, header_y, 'Error Category', fontsize=14, fontweight='bold', ha='center')
    ax.text(11, header_y, 'Sub-error Types', fontsize=14, fontweight='bold', ha='center')
    
    current_y = header_y - 0.5
    
    for i, category in enumerate(data):
        color = colors[i % len(colors)]
        
        # Alternating row colors
        row_color = '#fafafa' if i % 2 == 0 else 'white'
        
        # Full row background
        row_rect = Rectangle((0.5, current_y - row_height), 15, row_height,
                           facecolor=row_color, edgecolor='lightgray', linewidth=1)
        ax.add_patch(row_rect)
        
        # Category cell
        ax.text(0.7, current_y - 0.2, category['category'], 
                fontsize=12, fontweight='bold', color=color, va='top')
        
        # Sub-errors as comma-separated list
        sub_errors_text = ", ".join([se['error_type'] for se in category['sub_errors']])
        
        # Wrap text if too long
        if len(sub_errors_text) > 80:
            wrapped_text = '\n'.join(textwrap.wrap(sub_errors_text, width=80))
        else:
            wrapped_text = sub_errors_text
            
        ax.text(6.2, current_y - 0.2, wrapped_text, 
                fontsize=10, color='#333333', va='top')
        
        current_y -= row_height
    
    # Table borders
    ax.add_patch(Rectangle((0.5, current_y), 15, total_height - current_y - 1.0, 
                          fill=False, edgecolor='black', linewidth=2))
    
    # Vertical separator
    ax.axvline(x=6, ymin=current_y/(total_height), ymax=(total_height-1.0)/(total_height), 
               color='black', linewidth=1.5)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_three_column_table_landscape(data, output_file='nl2sql_taxonomy_3col_landscape.png'):
    """
    横屏优化版：自动根据总行数调整高度，列宽按横屏比例放大
    """
    colors = ['#dc2626', '#ea580c', '#d97706', '#16a34a', '#0891b2',
              '#7c3aed', '#be185d', '#059669', '#6366f1']

    # 横屏版列宽比例（单位是总宽度比例）
    fig_w = 28  # 横屏更宽
    left_ratio = 0.30
    mid_ratio = 0.25
    right_ratio = 0.45

    left_w = fig_w * left_ratio
    mid_w = fig_w * mid_ratio
    right_w = fig_w * right_ratio

    left_x = 0.5
    mid_x = left_x + left_w
    right_x = mid_x + mid_w
    table_w = left_w + mid_w + right_w

    row_h = 0.85
    line_h = 0.28
    total_sub = sum(len(c['sub_errors']) for c in data)
    base_top = 3.0
    base_bottom = 1.2
    total_h = base_top + total_sub * row_h + len(data) * 0.6 + base_bottom

    import textwrap
    fig, ax = plt.subplots(figsize=(fig_w, total_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, total_h)

    # 标题
    ax.text(fig_w / 2, total_h - 0.6,
            'A Taxonomy of NL2SQL Translation Semantic Errors',
            fontsize=22, fontweight='bold', ha='center', va='center')

    # 表头
    header_y = total_h - 1.6
    header_h = 0.8
    ax.add_patch(Rectangle((left_x, header_y), table_w, header_h,
                           facecolor='#eaeaea', edgecolor='black', linewidth=2))
    ax.text(left_x + left_w / 2, header_y + header_h / 2,
            'Major Error Category (with description)',
            fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(mid_x + mid_w / 2, header_y + header_h / 2,
            'Sub-error Name', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(right_x + right_w / 2, header_y + header_h / 2,
            'Sub-error Description', fontsize=14, fontweight='bold', ha='center', va='center')

    # 竖线
    ax.axvline(x=mid_x, color='black', linewidth=1.5)
    ax.axvline(x=right_x, color='black', linewidth=1.5)

    cur_y = header_y - 0.2
    for i, cat in enumerate(data):
        color = colors[i % len(colors)]
        sub_list = cat.get('sub_errors', [])

        # 左列文字换行宽度按列宽动态算
        wrap_left = int(left_w * 3.0)  # 数值可调
        wrap_right = int(right_w * 3.0)

        cat_desc = '\n'.join(textwrap.wrap(cat.get('description', ''), width=wrap_left))
        left_needed_h = 0.45 + max(1, len(cat_desc.split('\n'))) * line_h + 0.35
        sub_needed_h = max(row_h * max(1, len(sub_list)), 0.85)
        block_h = max(left_needed_h, sub_needed_h)

        ax.add_patch(Rectangle((left_x, cur_y - block_h), table_w, block_h,
                               facecolor=color, alpha=0.06, edgecolor=color, linewidth=1.5))
        ax.text(left_x + 0.25, cur_y - 0.25, cat['category'],
                fontsize=13, fontweight='bold', color=color, ha='left', va='top')
        ax.text(left_x + 0.25, cur_y - 0.25 - 0.45, cat_desc,
                fontsize=9.5, color='#444444', ha='left', va='top')

        if sub_list:
            top_y = cur_y - 0.2
            for j, se in enumerate(sub_list):
                y = top_y - j * row_h
                ax.text(mid_x + 0.25, y, se['error_type'],
                        fontsize=11, fontweight='bold', color='#222222',
                        ha='left', va='top')
                se_desc_wrapped = '\n'.join(textwrap.wrap(se.get('description', ''), width=wrap_right))
                ax.text(right_x + 0.25, y, se_desc_wrapped,
                        fontsize=9.5, color='#333333', ha='left', va='top')
        else:
            ax.text(mid_x + 0.25, cur_y - 0.25, '(no sub-errors)',
                    fontsize=10, color='#666666', ha='left', va='top')

        ax.axhline(y=cur_y - block_h - 0.02, color='lightgray', linewidth=0.8)
        cur_y -= (block_h + 0.6)

    ax.add_patch(Rectangle((left_x, 0.6), table_w, total_h - 2.2,
                           fill=False, edgecolor='black', linewidth=2))
    ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def create_detailed_table(data, output_file='nl2sql_taxonomy_detailed_table.png'):
    """Create a detailed table with descriptions."""
    
    colors = ['#dc2626', '#ea580c', '#d97706', '#16a34a', '#0891b2', 
              '#7c3aed', '#be185d', '#059669', '#6366f1']
    
    # Calculate total height needed
    total_sub_errors = sum(len(cat['sub_errors']) for cat in data)
    total_height = total_sub_errors * 0.8 + len(data) * 1.2 + 3
    
    fig, ax = plt.subplots(figsize=(20, total_height))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, total_height)
    
    # Title
    ax.text(10, total_height - 0.5, 'A Taxonomy of NL2SQL Translation Semantic Errors', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Table headers
    header_y = total_height - 1.2
    header_rect = Rectangle((0.5, header_y - 0.4), 19, 0.8,
                          facecolor='#e0e0e0', edgecolor='black', linewidth=2)
    ax.add_patch(header_rect)
    
    ax.text(3, header_y, 'Error Category', fontsize=14, fontweight='bold', ha='center')
    ax.text(8.5, header_y, 'Sub-error Type', fontsize=14, fontweight='bold', ha='center')
    ax.text(15, header_y, 'Description', fontsize=14, fontweight='bold', ha='center')
    
    current_y = header_y - 0.6
    
    for i, category in enumerate(data):
        color = colors[i % len(colors)]
        
        # Category section
        category_height = len(category['sub_errors']) * 0.8 + 0.4
        
        # Category background
        cat_rect = Rectangle((0.5, current_y - category_height), 19, category_height,
                           facecolor=color, alpha=0.1, edgecolor=color, linewidth=2)
        ax.add_patch(cat_rect)
        
        # Category name (spanning multiple rows)
        ax.text(1, current_y - 0.2, category['category'], 
                fontsize=13, fontweight='bold', color=color, va='top', rotation=0)
        
        # Sub-errors
        for j, sub_error in enumerate(category['sub_errors']):
            sub_y = current_y - 0.4 - j * 0.8
            
            # Sub-error name
            ax.text(6.5, sub_y, sub_error['error_type'], 
                    fontsize=11, fontweight='bold', color='#333333', va='center')
            
            # Description
            desc = sub_error['description'][:150] + "..." if len(sub_error['description']) > 150 else sub_error['description']
            wrapped_desc = '\n'.join(textwrap.wrap(desc, width=60))
            ax.text(11, sub_y, wrapped_desc, 
                    fontsize=9, color='#555555', va='center')
            
            # Row separator
            if j < len(category['sub_errors']) - 1:
                ax.axhline(y=sub_y - 0.4, color='lightgray', linewidth=0.5, alpha=0.7)
        
        current_y -= category_height
    
    # Table borders
    ax.add_patch(Rectangle((0.5, 0.2), 19, total_height - 1.5, 
                          fill=False, edgecolor='black', linewidth=2))
    
    # Vertical separators
    ax.axvline(x=6, ymin=0.2/total_height, ymax=(total_height-1.5)/total_height, 
               color='black', linewidth=1.5)
    ax.axvline(x=10.5, ymin=0.2/total_height, ymax=(total_height-1.5)/total_height, 
               color='black', linewidth=1.5)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# Main execution
if __name__ == "__main__":
    json_file = 'Spider/error_explaination.json'
    
    try:
        taxonomy_data = load_taxonomy_data(json_file)
        create_three_column_table_landscape(taxonomy_data)
        
        print("Creating simple table...")
        create_simple_table(taxonomy_data)
        
        print("Creating compact table...")
        create_compact_table(taxonomy_data)
        
        print("Creating detailed table...")
        create_detailed_table(taxonomy_data)
        
        print("All table visualizations created successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
        print("Please make sure the JSON file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")