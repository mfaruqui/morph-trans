CC=g++
CNN_DIR=$(CNN)
EIGEN_DIR=$(EIGEN)
BOOST_DIR=$(BOOST)

CNN_BUILD_DIR=$(CNN_DIR)/build
INCS=-I$(CNN_DIR) -I$(CNN_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(CNN_BUILD_DIR)/cnn/ -L$(BOOST_DIR)/lib
FINAL=-lcnn -lboost_regex -lboost_serialization -lboost_program_options -lrt -lpthread
CFLAGS=-std=c++11 -Ofast -g -march=native -pipe
BINDIR=bin
OBJDIR=obj
SRCDIR=src

.PHONY: clean
all: make_dirs $(BINDIR)/train-sep-morph $(BINDIR)/eval-ensemble-sep-morph $(BINDIR)/train-joint-enc-morph $(BINDIR)/eval-ensemble-joint-enc-morph $(BINDIR)/train-lm-sep-morph $(BINDIR)/eval-ensemble-lm-sep-morph $(BINDIR)/train-joint-enc-dec-morph $(BINDIR)/eval-ensemble-joint-enc-dec-morph $(BINDIR)/eval-ensemble-sep-morph-beam $(BINDIR)/train-lm-joint-enc $(BINDIR)/eval-ensemble-lm-joint-enc $(BINDIR)/eval-ensemble-joint-enc-beam $(BINDIR)/train-no-enc $(BINDIR)/eval-ensemble-no-enc $(BINDIR)/train-enc-dec $(BINDIR)/eval-ensemble-enc-dec $(BINDIR)/train-enc-dec-attn $(BINDIR)/eval-ensemble-enc-dec-attn

make_dirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

include $(wildcard $(OBJDIR)/*.d)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@
	$(CC) -MM -MP -MT "$@" $(CFLAGS) $(INCS) $< > $(OBJDIR)/$*.d

$(BINDIR)/train-sep-morph: $(addprefix $(OBJDIR)/, train-sep-morph.o sep-morph.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train-no-enc: $(addprefix $(OBJDIR)/, train-no-enc.o no-enc.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train-enc-dec: $(addprefix $(OBJDIR)/, train-enc-dec.o enc-dec.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train-enc-dec-attn: $(addprefix $(OBJDIR)/, train-enc-dec-attn.o enc-dec-attn.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train-joint-enc-morph: $(addprefix $(OBJDIR)/, train-joint-enc-morph.o joint-enc-morph.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train-joint-enc-dec-morph: $(addprefix $(OBJDIR)/, train-joint-enc-dec-morph.o joint-enc-dec-morph.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train-lm-sep-morph: $(addprefix $(OBJDIR)/, train-lm-sep-morph.o lm-sep-morph.o utils.o lm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/train-lm-joint-enc: $(addprefix $(OBJDIR)/, train-lm-joint-enc.o lm-joint-enc.o utils.o lm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-lm-joint-enc: $(addprefix $(OBJDIR)/, eval-ensemble-lm-joint-enc.o utils.o lm-joint-enc.o lm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-sep-morph: $(addprefix $(OBJDIR)/, eval-ensemble-sep-morph.o utils.o sep-morph.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-no-enc: $(addprefix $(OBJDIR)/, eval-ensemble-no-enc.o utils.o no-enc.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-enc-dec: $(addprefix $(OBJDIR)/, eval-ensemble-enc-dec.o utils.o enc-dec.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-enc-dec-attn: $(addprefix $(OBJDIR)/, eval-ensemble-enc-dec-attn.o utils.o enc-dec-attn.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-lm-sep-morph: $(addprefix $(OBJDIR)/, eval-ensemble-lm-sep-morph.o utils.o lm-sep-morph.o lm.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-joint-enc-morph: $(addprefix $(OBJDIR)/, eval-ensemble-joint-enc-morph.o utils.o joint-enc-morph.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-joint-enc-dec-morph: $(addprefix $(OBJDIR)/, eval-ensemble-joint-enc-dec-morph.o utils.o joint-enc-dec-morph.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-sep-morph-beam: $(addprefix $(OBJDIR)/, eval-ensemble-sep-morph-beam.o utils.o sep-morph.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/eval-ensemble-joint-enc-beam: $(addprefix $(OBJDIR)/, eval-ensemble-joint-enc-beam.o utils.o joint-enc-morph.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)


clean:
	rm -rf $(BINDIR)/*
	rm -rf $(OBJDIR)/*
