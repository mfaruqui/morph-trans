CC=g++
CNN_DIR = ../cnn
EIGEN = /opt/tools/eigen-dev/
CNN_BUILD_DIR=$(CNN_DIR)/build
INCS=-I$(CNN_DIR) -I$(CNN_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(CNN_BUILD_DIR)/cnn/ -L/opt/tools/boost_1_55_0/lib
FINAL=-lcnn -lboost_regex -lboost_serialization -lboost_program_options -lrt -lpthread
CFLAGS=-std=c++11 -Ofast -g -march=native -pipe
BINDIR=bin
OBJDIR=obj
SRCDIR=src

.PHONY: clean
all: make_dirs $(BINDIR)/train-sep-morph $(BINDIR)/eval-ensemble-sep-morph $(BINDIR)/train-joint-enc-morph $(BINDIR)/eval-ensemble-joint-enc-morph $(BINDIR)/train-lm-sep-morph $(BINDIR)/eval-ensemble-lm-sep-morph $(BINDIR)/train-joint-enc-dec-morph $(BINDIR)/eval-ensemble-joint-enc-dec-morph

make_dirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

include $(wildcard $(OBJDIR)/*.d)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@
	$(CC) -MM -MP -MT "$@" $(CFLAGS) $(INCS) $< > $(OBJDIR)/$*.d

$(BINDIR)/train-sep-morph: $(addprefix $(OBJDIR)/, train-sep-morph.o sep-morph.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train-joint-enc-morph: $(addprefix $(OBJDIR)/, train-joint-enc-morph.o joint-enc-morph.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train-joint-enc-dec-morph: $(addprefix $(OBJDIR)/, train-joint-enc-dec-morph.o joint-enc-dec-morph.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train-lm-sep-morph: $(addprefix $(OBJDIR)/, train-lm-sep-morph.o lm-sep-morph.o utils.o lm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-sep-morph: $(addprefix $(OBJDIR)/, eval-ensemble-sep-morph.o utils.o sep-morph.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-lm-sep-morph: $(addprefix $(OBJDIR)/, eval-ensemble-lm-sep-morph.o utils.o lm-sep-morph.o lm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-joint-enc-morph: $(addprefix $(OBJDIR)/, eval-ensemble-joint-enc-morph.o utils.o joint-enc-morph.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-joint-enc-dec-morph: $(addprefix $(OBJDIR)/, eval-ensemble-joint-enc-dec-morph.o utils.o joint-enc-dec-morph.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)


clean:
	rm -rf $(BINDIR)/*
	rm -rf $(OBJDIR)/*
