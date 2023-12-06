Enda Hargaden, July 2016
UT Economics/CBER
Feel free to distribute
The file "beamerthemeutk.sty" makes a beamer theme that matches the official University of Tennessee Powerpoint template


============================
How to install the UTK theme
============================

1. Place "beamerthemeutk.sty" into your beamer theme folder. This folder is somewhat obscure. For me, using MiKTeX to compile the document, it's at
C:\Program Files\MiKTeX 2.9\tex\latex\beamer\base\themes\theme

2. Make sure TeX knows you have placed the theme in there. For me, that requires "Refreshing the Filename Database (FNDB)" in MiKTeX settings. I access this by opening the settings up from Start > MiKTeX > Maintenance.

3. Place both "utkbeamerbottom.png" and "utkbeamerfront.png" into the same folder as your lecture slides. Yes, this is an inelegant solution that will require repetition. If someone knows a better solution, please let me know.

4. Use the following basic template for your slides:

\documentclass{beamer}
\usetheme{utk}

\title[Optional Short Title for Footer]{My Title}
\author{My Name}
\institute{My Workplace}

\begin{document}

\begin{frame}
\titlepage
\end{frame}

\begin{frame}
\frametitle{This is my title}
And this is my content
\end{frame}

\end{document}
