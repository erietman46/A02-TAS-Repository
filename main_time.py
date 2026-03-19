# call all over functions, make everything look nice, ...
from zero_crossings import zero_crossings
from total_variation import total_variation
from error_pdf import error_pdf
import pandas as pd
from openpyxl.styles import PatternFill

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

'''Zero-crossings of e'''
data, stat_res = zero_crossings()
print(f"\nMean Zero-crossings: \n{data}")
print(f"\nStatistical Results: \n{stat_res}\n")

'''Total Variation of u'''
data2, stat_res2 = total_variation()
print(f"\nMean Total Variation: \n{data2}")
print(f"\nStatistical Results: \n{stat_res2}\n")

'''Mean and 1-sigma Interval of e'''
data3, stat_res3, data4, stat_res4 = error_pdf()
print(f"\nMean 1-sigma Interval Width: \n{data4}")
print(f"\nStatistical Results: \n{stat_res4}\n")

'''Output an Excel file'''
highlight = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

with pd.ExcelWriter("results.xlsx", engine="openpyxl") as writer:
    # === Zero-crossings of e ===
    data, stat_res = zero_crossings()

    data.to_excel(writer, sheet_name="ZeroCrossings_data")
    stat_res.to_excel(writer, sheet_name="ZeroCrossings_stats")

    sheet = writer.sheets["ZeroCrossings_stats"]

    # Highlight p-values (first column)
    for row in sheet.iter_rows(min_row=2):
        cell = row[1]
        if float(cell.value) < 0.05:
            cell.fill = highlight

    # === Total Variation of u ===
    data2, stat_res2 = total_variation()

    data2.to_excel(writer, sheet_name="TotalVariation_data")
    stat_res2.to_excel(writer, sheet_name="TotalVariation_stats")

    sheet = writer.sheets["TotalVariation_stats"]

    for row in sheet.iter_rows(min_row=2):
        cell = row[1]
        if float(cell.value) < 0.05:
            cell.fill = highlight

    # === Mean and 1-sigma Interval of e ===
    data3, stat_res3, data4, stat_res4 = error_pdf()

    data4.to_excel(writer, sheet_name="ErrorPDF_data")
    stat_res4.to_excel(writer, sheet_name="ErrorPDF_stats")

    sheet = writer.sheets["ErrorPDF_stats"]

    for row in sheet.iter_rows(min_row=2):
        cell = row[1]
        if float(cell.value) < 0.05:
            cell.fill = highlight

'''If a .txt file with the outputs are more useful, use this
with open("results.txt", "w") as f:
    # Zero-crossings of e
    f.write("=== Zero-crossings of e ===\n")
    data, stat_res = zero_crossings()

    f.write("\nMean Zero-crossings:\n")
    f.write(data.to_string())
    f.write("\n\nStatistical Results:\n")
    f.write(stat_res.to_string())
    f.write("\n\n")

    # Total Variation of u
    f.write("=== Total Variation of u ===\n")
    data2, stat_res2 = total_variation()

    f.write("\nMean Total Variation:\n")
    f.write(data2.to_string())
    f.write("\n\nStatistical Results:\n")
    f.write(stat_res2.to_string())
    f.write("\n\n")

    # Mean and 1-sigma Interval of e
    f.write("=== Mean and 1-sigma Interval of e ===\n")
    data3, stat_res3, data4, stat_res4 = error_pdf()

    f.write("\nMean 1-sigma Interval Width:\n")
    f.write(data4.to_string())
    f.write("\n\nStatistical Results:\n")
    f.write(stat_res4.to_string())
    f.write("\n")
'''