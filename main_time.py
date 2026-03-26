# call all over functions, make everything look nice, ...
from zero_crossings import zero_crossings
from total_variation import total_variation
from error_pdf import error_pdf
from RMS_error import RMS_error
from RMS_u import RMS_input
from derivative_e_RMS import RMS_DERe
from derivative_u_RMS import RMS_DERu
from contributions import contributions
import pandas as pd
from openpyxl.styles import PatternFill

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

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

    # === RMS of e ===
    data5, stat_res5 = RMS_error()

    data5.to_excel(writer, sheet_name="RMSe_data")
    stat_res5.to_excel(writer, sheet_name="RMSe_stats")

    sheet = writer.sheets["RMSe_stats"]

    for row in sheet.iter_rows(min_row=2):
        cell = row[1]
        if float(cell.value) < 0.05:
            cell.fill = highlight

    # === RMS of u ===
    data6, stat_res6 = RMS_input()

    data6.to_excel(writer, sheet_name="RMSu_data")
    stat_res6.to_excel(writer, sheet_name="RMSu_stats")

    sheet = writer.sheets["RMSu_stats"]

    for row in sheet.iter_rows(min_row=2):
        cell = row[1]
        if float(cell.value) < 0.05:
            cell.fill = highlight

    # === RMS of DERe ===
    data7, stat_res7 = RMS_DERe()

    data7.to_excel(writer, sheet_name="RMSDERe_data")
    stat_res7.to_excel(writer, sheet_name="RMSDERe_stats")

    sheet = writer.sheets["RMSDERe_stats"]

    for row in sheet.iter_rows(min_row=2):
        cell = row[1]
        if float(cell.value) < 0.05:
            cell.fill = highlight

    # === RMS of DERu ===
    data8, stat_res8 = RMS_DERu()

    data8.to_excel(writer, sheet_name="RMSDERu_data")
    stat_res8.to_excel(writer, sheet_name="RMSDERu_stats")

    sheet = writer.sheets["RMSDERu_stats"]

    for row in sheet.iter_rows(min_row=2):
        cell = row[1]
        if float(cell.value) < 0.05:
            cell.fill = highlight

    # === Contributions to u ===
    data9, stat_res9, data10, stat_res10, data11, stat_res11 = contributions()

    # --- WRITE DATA SHEET ---
    start_row = 0

    # f_d contribution
    data9.to_excel(writer, sheet_name="Contrib_data", startrow=start_row)
    start_row += len(data9) + 3

    # f_t contribution
    data10.to_excel(writer, sheet_name="Contrib_data", startrow=start_row)
    start_row += len(data10) + 3

    # noise contribution
    data11.to_excel(writer, sheet_name="Contrib_data", startrow=start_row)

    # --- WRITE STATS SHEET ---
    start_row = 0

    # f_d stats
    stat_res9.to_excel(writer, sheet_name="Contrib_stats", startrow=start_row)
    start_row += len(stat_res9) + 3

    # f_t stats
    stat_res10.to_excel(writer, sheet_name="Contrib_stats", startrow=start_row)
    start_row += len(stat_res10) + 3

    # noise stats
    stat_res11.to_excel(writer, sheet_name="Contrib_stats", startrow=start_row)

    # --- APPLY HIGHLIGHTING ---
    sheet = writer.sheets["Contrib_stats"]

    current_row = 0
    for stat_res in [stat_res9, stat_res10, stat_res11]:

        for row in sheet.iter_rows(min_row=current_row + 2,
                                   max_row=current_row + 1 + len(stat_res)):

            cell = row[1]  # p-value column
            if float(cell.value) < 0.05:
                cell.fill = highlight

        current_row += len(stat_res) + 3

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