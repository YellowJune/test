\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{vanish_ieee}[2026/08/08 IEEE-compatible two-column research format]
\LoadClass[10pt,twocolumn,letterpaper]{article}
\RequirePackage[letterpaper,top=0.70in,bottom=0.72in,left=0.63in,right=0.63in,columnsep=0.24in]{geometry}
\RequirePackage{mathptmx}
\RequirePackage{microtype}
\RequirePackage{titlesec}
\RequirePackage{fancyhdr}
\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}
\setlength{\columnsep}{0.24in}
\titleformat{\section}{\normalfont\bfseries\scshape\centering}{\Roman{section}.}{0.45em}{}
\titleformat{\subsection}{\normalfont\bfseries\itshape}{\Alph{subsection}.}{0.45em}{}
\titleformat{\subsubsection}{\normalfont\itshape}{\arabic{subsubsection})}{0.4em}{}
\titlespacing*{\section}{0pt}{1.25ex plus .4ex minus .2ex}{0.7ex}
\titlespacing*{\subsection}{0pt}{1.0ex plus .3ex minus .2ex}{0.45ex}
\titlespacing*{\subsubsection}{0pt}{0.8ex plus .2ex minus .2ex}{0.35ex}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\thepage}
\setlength{\footskip}{0.33in}
\renewcommand{\maketitle}{%
  \twocolumn[\begin{@twocolumnfalse}
  \begin{center}
  {\LARGE\bfseries \@title\par}\vspace{0.65em}
  {\large \@author\par}\vspace{0.5em}
  \end{center}
  \end{@twocolumnfalse}\vspace{0.5em}]}
\AtBeginDocument{\sloppy}
