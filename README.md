# UDGraphs: Unit Distance Graphs

The folder distribution must remain as is, otherwise
the app won't work.

This app can:
	- Build the different graphs used by de Grey and Heule.
	- Color them.
	- Draw them using LaTeX, and opening the pdf file.

It uses different Python libraries, that can be found in 'requirements.txt'. There's a Windows executable version available as well.

-- Warning! --

In order to draw the different graphs, this app makes use of the
"pdflatex" and/or the "lualatex" commands.

Thus, if you haven't got LaTeX installed, the draw function
will only generate the .tex files, but it won't compile them.
