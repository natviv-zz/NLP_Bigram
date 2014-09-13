package nlp.lm;

import java.io.*;
import java.util.*;

/** 
 * @author Ray Mooney
 * A simple bigram language model that uses simple fixed-weight interpolation
 * with a unigram model for smoothing.
*/

public class BidirectionalModel {

    
    public BidirectionalModel() {
	
    }

    /** Use sentences as a test set to evaluate the model. Print out perplexity
     *  of the model for this test data */
    public void test (List<List<String>> sentences, BigramModel model, BackwardBigramModel rmodel) {
	// Compute log probability of sentence to avoid underflow
	double totalLogProb = 0;
	// Keep count of total number of tokens predicted
	double totalNumTokens = 0;
	// Accumulate log prob of all test sentences
	for (List<String> sentence : sentences) {
	    // Num of tokens in sentence plus 1 for predicting </S>
	    totalNumTokens += sentence.size() + 1;
	    // Compute log prob of sentence
	    double sentenceLogProbf = 0.2*model.sentenceLogProb(sentence);
	    
	    Collections.reverse(sentence);
	    
	    double sentenceLogProbr = 0.8*rmodel.sentenceLogProb(sentence);
	    
	    //	    System.out.println(sentenceLogProb + " : " + sentence);
	    // Add to total log prob (since add logs to multiply probs)
	    totalLogProb = totalLogProb + sentenceLogProbf + sentenceLogProbr;
	}
	
	// Given log prob compute perplexity
	double perplexity = Math.exp(-totalLogProb / totalNumTokens);
	System.out.println("Perplexity = " + perplexity );
    }
    
    /* Compute log probability of sentence given current model */
   
    public void test2 (List<List<String>> sentences, BigramModel fmodel, BackwardBigramModel rmodel) {
    	double totalLogProb = 0;
    	double totalNumTokens = 0;
    	for (List<String> sentence : sentences) {
    	    totalNumTokens += sentence.size();
    	    double sentenceLogProb = sentenceLogProb2(sentence,fmodel,rmodel);
    	    //	    System.out.println(sentenceLogProb + " : " + sentence);
    	    totalLogProb += sentenceLogProb;
    	}
    	double perplexity = Math.exp(-totalLogProb / totalNumTokens);
    	System.out.println("Word Perplexity = " + perplexity );
      
        }
        
        /** Like sentenceLogProb but excludes predicting end-of-sentence when computing prob */
        public double sentenceLogProb2 (List<String> sentence, BigramModel fmodel, BackwardBigramModel rmodel) {
    	/*String prevToken = "<S>";
    	double sentenceLogProb = 0;
    	for (String token : sentence) {
    	    DoubleValue unigramVal = fmodel.unigramMap.get(token);
    	    if (unigramVal == null) {
    		token = "<UNK>";
    		unigramVal = fmodel.unigramMap.get(token);
    	    }
    	    String bigram = fmodel.bigram(prevToken, token);
    	    DoubleValue bigramVal = fmodel.bigramMap.get(bigram);
    	    double logProb = Math.log(fmodel.interpolatedProb(unigramVal, bigramVal));
    	    sentenceLogProb += logProb;
    	    prevToken = token;    	    
    	}
    	*/
    	 String prevToken = "<S>";
    	 double sentenceLogProb = 0;
    	 String nextToken;
    	//double sentenceLogProb = 0;
    	for (int i = 0; i < sentence.size(); i++) {
    		String token = sentence.get(i);
    		if(i == (sentence.size()-1))
    			nextToken = "</S>";
    		else
    			nextToken = sentence.get(i+1);
    		
    	    DoubleValue unigramVal = fmodel.unigramMap.get(token);
    	    if (unigramVal == null) {
    		token = "<UNK>";
    		unigramVal = fmodel.unigramMap.get(token);
    	    }
    	    DoubleValue unigramVal2 = fmodel.unigramMap.get(nextToken);
    	    if (unigramVal2 == null) {
    		nextToken = "<UNK>";
    		unigramVal2 = fmodel.unigramMap.get(nextToken);
    	    }
    	    
    	    
    	    String bigram = fmodel.bigram(prevToken, token);
    	    DoubleValue bigramVal1 = fmodel.bigramMap.get(bigram);
    	    if(bigramVal1 == null)
    	    	bigramVal1 = new DoubleValue();
    	    bigram = fmodel.bigram(nextToken, token);
    	    DoubleValue bigramVal2 = rmodel.bigramMap.get(bigram);
    	    if(bigramVal2 == null)
    	    	bigramVal2 = new DoubleValue();
    	    DoubleValue bigramVal = new DoubleValue();
    	    //bigramVal.setValue((bigramVal1.getValue()+bigramVal2.getValue())/2);
    	    bigramVal.setValue(0.6*bigramVal1.getValue()+0.4*bigramVal2.getValue());
    	    double logProb = Math.log(rmodel.interpolatedProb(unigramVal, bigramVal));
    	    sentenceLogProb += logProb;
    	    prevToken = token;
    	}
    		
    	return sentenceLogProb;
        }


   

    public static int wordCount (List<List<String>> sentences) {
    	int wordCount = 0;
    	for (List<String> sentence : sentences) {
    	    wordCount += sentence.size();
    	}
    	return wordCount;
        }
   

    /** Train and test a bigram model.
     *  Command format: "nlp.lm.BigramModel [DIR]* [TestFrac]" where DIR 
     *  is the name of a file or directory whose LDC POS Tagged files should be 
     *  used for input data; and TestFrac is the fraction of the sentences
     *  in this data that should be used for testing, the rest for training.
     *  0 < TestFrac < 1
     *  Uses the last fraction of the data for testing and the first part
     *  for training.
     */
    
    public static void main(String[] args) throws IOException {
	// All but last arg is a file/directory of LDC tagged input data
	File[] files = new File[args.length - 1];
	for (int i = 0; i < files.length; i++) 
	    files[i] = new File(args[i]);
	// Last arg is the TestFrac
	double testFraction = Double.valueOf(args[args.length -1]);
	// Get list of sentences from the LDC POS tagged input files
	List<List<String>> sentences = 	POSTaggedFile.convertToTokenLists(files);
	int numSentences = sentences.size();
	// Compute number of test sentences based on TestFrac
	int numTest = (int)Math.round(numSentences * testFraction);
	// Take test sentences from end of data
	List<List<String>> testSentences = sentences.subList(numSentences - numTest, numSentences);
	// Take training sentences from start of data
	List<List<String>> trainSentences = sentences.subList(0, numSentences - numTest);
	System.out.println("Training forward model");
	System.out.println("# Train Sentences = " + trainSentences.size() + 
			   " (# words = " + wordCount(trainSentences) + 
			   ") \n# Test Sentences = " + testSentences.size() +
			   " (# words = " + wordCount(testSentences) + ")");
	// Create a bigram model and train it.
	BigramModel model = new BigramModel();
	System.out.println("Training...");
	model.train(trainSentences);
	// Test on training data using test and test2
	model.test(trainSentences);
	model.test2(trainSentences);
	System.out.println("Testing...");
	// Test on test data using test and test2
	model.test(testSentences);
	model.test2(testSentences);
	//model.print();
	
	for(List<String> sentence : sentences)
	{
		Collections.reverse(sentence);
	}
	
	
	numSentences = sentences.size();
	// Compute number of test sentences based on TestFrac
	numTest = (int)Math.round(numSentences * testFraction);
	// Take test sentences from end of data
	testSentences = sentences.subList(numSentences - numTest, numSentences);
	// Take training sentences from start of data
	trainSentences = sentences.subList(0, numSentences - numTest);
	
	System.out.println("Training backward model");
	
	
	
	BackwardBigramModel rmodel = new BackwardBigramModel();
	System.out.println("Training...");
	rmodel.train(trainSentences);
	// Test on training data using test and test2
	rmodel.test(trainSentences);
	rmodel.test2(trainSentences);
	System.out.println("Testing...");
	// Test on test data using test and test2
	rmodel.test(testSentences);
	rmodel.test2(testSentences);
	//rmodel.print();
	System.out.println("Testing bidirectional model");
	BidirectionalModel bmodel = new BidirectionalModel();
	System.out.println("Training...");
	bmodel.test(trainSentences,model,rmodel);
	bmodel.test2(trainSentences,model,rmodel);
	System.out.println("Testing...");
	bmodel.test(testSentences,model,rmodel);
	bmodel.test2(testSentences,model,rmodel);
	
	
	
	
    }

}
