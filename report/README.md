# Public Technical Report

`main.tex` is the sanitized public report source. It identifies the authors as
Hannah Abedin and Erfan Razmand and contains only their public names.

The report uses 12-point Times-style text, one-inch margins, 1.15 line spacing,
and a single-column academic layout with equations, tables, figures, and
references.

Compile from the `report/` directory so the relative figure paths resolve:

```bash
tectonic -o ../docs main.tex
mv ../docs/main.pdf ../docs/Forecast_Driven_Elevator_Control_Report.pdf
```

The reviewed PDF in `docs/` is the public portfolio edition.
