
DOT = dot
DOTOPTS = -Tps

TEX = latex

#-----------------------------
all: manual devel

debuggerChart: debug_gpu.dot
	dot -Tps debug_gpu.dot  > debug_gpu.ps

flowchart:
	dot -Tps flowchart.dot > flowchart.ps

manual: manual.tex
	$(TEX) manual.tex
	dvipdf manual.dvi

devel: devel.tex imogenInvoke.ps
	latex devel.tex
	dvipdf devel.dvi

cleanup:
	rm -f *aux *dvi *log *out

imogenInvoke.ps: imogenInvoke.dot
	$(DOT) $(DOTOPTS) imogenInvoke.dot -o imogenInvoke.ps
