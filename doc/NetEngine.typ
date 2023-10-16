#let Blue = rgb("#127bca")
#let TextGrey = rgb("#32323C")

#set text(
    font: "Open Sans",
    weight: "regular",
    size: 10pt
)

#let title = "NetEngine"

#set page(
  header: rect(
    stroke: (bottom: (paint: Blue, thickness: 0.1pt)),
    inset: (x: 0pt, top: 0pt, bottom: 1.5em),
    grid(
        columns: (auto, 1fr, 20%),
        rows: (10mm),
        text(fill: Blue, title),
        "", 
    )
  ),
  footer: rect(
    stroke: (top: (paint: Blue, thickness: 0.1pt)),
    inset: (x: 0pt, bottom: 0pt, top: 1em),
    text(TextGrey.lighten(50%))[#h(1fr) Page #counter(page).display()]
  ),
  margin: (x: 25mm, y: 30mm)
)

= Hello

Test @RUBERT2018423.

$ a/b*C $

#pagebreak()

= Hello

Test.

$ a/b*C $

#pagebreak()

#bibliography("bibliography.bib") 