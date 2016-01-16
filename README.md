# morph-trans
Manaal Faruqui, manaalfar@gmail.com

This tool is used to generate morphologically inflected forms of a given word according to a specified morphological attribute. The input word is treated as a sequence of characters and the output word is generated as the transofrmed sequence of characters using a variant of neural encoder-decoder architecture. This is described in Faruqui et al (2016).

###Requirements

1. CNN neural network library: https://github.com/clab/cnn
2. C++ BOOST library: http://www.boost.org/
3. C++ Eigen library: http://eigen.tuxfamily.org/

Please download and compile these libraries.

###Data

You need three files to train the inflection generation system. Two files that contain vocabulary of the characters, and the vocabulary of the morphological attributes. The third file contains the training/dev/test data.

1. Character vocabulary file: char_vocab.txt

```<s> </s> a b c g f ä ...```

2. Morphological attribute file: morph_vocab.txt

```case=nominative:number=singular case=dative:number=plural ...```

3. Inflection train/dev/test file: infl.txt

```<s> a a l </s>|<s> a a l e s </s>|case=genitive:number=singular```
