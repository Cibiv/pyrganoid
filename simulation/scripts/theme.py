def theme(p, font_size=5):
    p.border_fill_color = None
    p.background_fill_color = None
    p.outline_line_color = None
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.major_label_text_font = "helvetica"
    p.yaxis.major_label_text_font = "helvetica"
    p.xaxis.major_label_text_font_size = f"{font_size}pt"
    p.yaxis.major_label_text_font_size = f"{font_size}pt"

    p.xaxis.axis_label_text_font = "helvetica"
    p.xaxis.axis_label_text_font_size = f"{font_size}pt"
    p.yaxis.axis_label_text_font = "helvetica"
    p.yaxis.axis_label_text_font_size = f"{font_size}pt"
    p.legend.label_text_font = "helvetica"
    p.legend.label_text_font_size = f"{font_size}pt"

    p.xaxis.axis_line_width = 0.66
    p.yaxis.axis_line_width = 0.66

    p.xaxis.major_tick_line_width = 0.66
    p.yaxis.major_tick_line_width = 0.66

    p.xaxis.major_tick_in = 2
    p.xaxis.major_tick_out = 2
    p.yaxis.major_tick_in = 2
    p.yaxis.major_tick_out = 2

    return p


