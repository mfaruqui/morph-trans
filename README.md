# morph-trans
Manaal Faruqui, manaalfar@gmail.com

This tool is used to generate morphologically inflected forms of a given word according to a specified morphological attribute. The input word is treated as a sequence of characters and the output word is generated as the transofrmed sequence of characters using a variant of neural encoder-decoder architecture. This is described in Faruqui et al (NAACL, 2016).

###Requirements

1. CNN neural network library: https://github.com/clab/cnn
2. C++ BOOST library: http://www.boost.org/
3. C++ Eigen library: http://eigen.tuxfamily.org/

Please download and compile these libraries.

###Data

You need three files to train the inflection generation system. Two files that contain vocabulary of the characters, and the vocabulary of the morphological attributes. The third file contains the training/dev/test data. Sample files can be found in data/ .

* Character vocabulary file: char_vocab.txt

```<s> </s> a b c g f ä ...```

* Morphological attribute file: morph_vocab.txt

```case=nominative:number=singular case=dative:number=plural ...```

* Inflection train/dev/test file: infl.txt

```<s> a a l </s>|<s> a a l e s </s>|case=genitive:number=singular```

* Optional language models: [here] (https://drive.google.com/folderview?id=0B93-ltInuGUyeGhDUVZSeWkyUVE&usp=sharing)

###Compile

After you have installed, CNN, Boost and Eige, you can simply install the software by typing the following command:-

```make CNN=cnn-dir BOOST=boost-dir EIGEN=eigen-dir```

###Run

To train the inflection generation system, simply run the following:-

```./bin/train-sep-morph char_vocab.txt morph_vocab.txt train_infl.txt dev_infl.txt 100 30 1e-5 1 model.txt```

Here, 100 is the hidden layer size of the LSTM, 30 is the number of iterations for training, 1e-5 is the l2 regularization strength and 1 is the number of layers in the LSTM.

To test the system, run:-

```./bin/eval-ensemble-sep-morph char_vocab.txt morph_vocab.txt test_infl.txt model1.txt model2.txt model3.txt ... > output.txt```

This can use an ensemble of models for evaluation. If you want to use only one model, just provide one model. The sep-moprh model is the model that provided us best supervised results. Other models can be used in the same way. Baseline encoder-decoder models can be trained using ```train-enc-dec``` and ```train-enc-dec-attn``` models.

###Reference
```
@inproceedings{faruqui:2016:infl,
  author    = {Faruqui, Manaal and Tsvetkov, Yulia and Neubig, Graham and Dyer, Chris},
  title     = {Morphological Inflection Generation Using Character Sequence to Sequence Learning},
  journal = {Proc. of NAACL},
  year      = {2016},
}
```
