CC=g++
CNN_DIR = ../cnn
EIGEN = /opt/tools/eigen-dev/
CNN_BUILD_DIR=$(CNN_DIR)/build
INCS=-I$(CNN_DIR) -I$(CNN_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(CNN_BUILD_DIR)/cnn/
FINAL=-lcnn -lboost_regex -lboost_serialization -lboost_program_options -lrt -lpthread
#FINAL=-lcnn -lcnncuda -lboost_regex -lboost_serialization -lboost_program_options -lcuda -lcudart -lcublas
CFLAGS=-std=c++11 -Ofast -g -march=native -pipe
#CFLAGS=-std=c++11 -Wall -pedantic -O0 -g -pipe
BINDIR=bin
OBJDIR=obj
SRCDIR=src

.PHONY: clean
all: make_dirs $(BINDIR)/train-sep-morph $(BINDIR)/eval-sep-morph $(BINDIR)/eval-ensemble-sep-morph

make_dirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

include $(wildcard $(OBJDIR)/*.d)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@
	$(CC) -MM -MP -MT "$@" $(CFLAGS) $(INCS) $< > $(OBJDIR)/$*.d

$(BINDIR)/train-sep-morph: $(addprefix $(OBJDIR)/, train-sep-morph.o sep-morph.o utils.o  decode.o read-write.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-sep-morph: $(addprefix $(OBJDIR)/, eval-sep-morph.o utils.o sep-morph.o decode.o read-write.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-sep-morph: $(addprefix $(OBJDIR)/, eval-ensemble-sep-morph.o utils.o sep-morph.o decode.o read-write.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

clean:
	rm -rf $(BINDIR)/*
	rm -rf $(OBJDIR)/*
