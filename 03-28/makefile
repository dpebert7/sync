GPP = gcc
OUTPUT = temp2
LDFLAGS = -lGL -lGLU -lglut -lm

UNAME_S := $(shell uname -s)
ifeq "$(UNAME_S)" "Darwin"
	LDFLAGS =-framework OpenGL -framework GLUT
endif

$(OUTPUT): mj_forces2.c
	$(GPP) -o $@ $< $(LDFLAGS)