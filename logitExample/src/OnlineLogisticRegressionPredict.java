package logitExample.src;

import com.google.common.base.Charsets;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import com.google.common.io.Resources;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.commons.csv.CSVUtils;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.ContinuousValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;
import org.apache.mahout.vectorizer.encoders.TextValueEncoder;
import org.apache.mahout.classifier.evaluation.Auc;
import org.apache.mahout.classifier.sgd.CsvRecordFactory;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.LogisticModelParameters;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.classifier.sgd.TrainLogistic;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public class OnlineLogisticRegressionPredict {
	private static String modelFile="/home/ctiwary/logistic/model";
	private static String inputFileName="/home/ctiwary/bank-additional-full.csv";
	private static Map<Integer, FeatureVectorEncoder> predictorEncoders;
	private static List<Integer> predictors;
	private static Map typeMap;
	private static boolean includeBiasTerm;
	private static final String CANNOT_CONSTRUCT_CONVERTER =
		      "Unable to construct type converter... shouldn't be possible";
	
	private static final Map<String, Set<Integer>> traceDictionary = Maps.newTreeMap();
	
	private static final String INTERCEPT_TERM = "Intercept Term";

	private static final Map<String, Class<? extends FeatureVectorEncoder>> TYPE_DICTIONARY =
	          ImmutableMap.<String, Class<? extends FeatureVectorEncoder>>builder()
	                  .put("continuous", ContinuousValueEncoder.class)
	                  .put("numeric", ContinuousValueEncoder.class)
	                  .put("n", ContinuousValueEncoder.class)
	                  .put("word", StaticWordValueEncoder.class)
	                  .put("w", StaticWordValueEncoder.class)
	                  .put("text", TextValueEncoder.class)
	                  .put("t", TextValueEncoder.class)
	                  .build();
	
	public static void firstLine(String line) {
	    // read variable names, build map of name -> column
	    final Map<String, Integer> vars = Maps.newHashMap();
	    List<String> variableNames = parseCsvLine(line);
	    int column = 0;
	    for (String var : variableNames) {
	      vars.put(var, column++);
	    }


	    

	    // create list of predictor column numbers
	    predictors = Lists.newArrayList(Collections2.transform(typeMap.keySet(), new Function<String, Integer>() {
	      public Integer apply(String from) {
	        Integer r = vars.get(from);
	        Preconditions.checkArgument(r != null, "Can't find variable %s, only know about %s", from, vars);
	        return r;
	      }
	    }));

	    if (includeBiasTerm) {
	      predictors.add(-1);
	    }
	    Collections.sort(predictors);

	    // and map from column number to type encoder for each column that is a predictor
	    predictorEncoders = Maps.newHashMap();
	    for (Integer predictor : predictors) {
	      String name;
	      Class<? extends FeatureVectorEncoder> c;
	      if (predictor == -1) {
	        name = INTERCEPT_TERM;
	        c = ConstantValueEncoder.class;
	      } else {
	        name = variableNames.get(predictor);
	        c = TYPE_DICTIONARY.get(typeMap.get(name));
	      }
	      try {
	        Preconditions.checkArgument(c != null, "Invalid type of variable %s,  wanted one of %s",
	          typeMap.get(name), TYPE_DICTIONARY.keySet());
	        Constructor<? extends FeatureVectorEncoder> constructor = c.getConstructor(String.class);
	        Preconditions.checkArgument(constructor != null, "Can't find correct constructor for %s", typeMap.get(name));
	        FeatureVectorEncoder encoder = constructor.newInstance(name);
	        predictorEncoders.put(predictor, encoder);
	        encoder.setTraceDictionary(traceDictionary);
	      } catch (InstantiationException e) {
	        throw new IllegalStateException(CANNOT_CONSTRUCT_CONVERTER, e);
	      } catch (IllegalAccessException e) {
	        throw new IllegalStateException(CANNOT_CONSTRUCT_CONVERTER, e);
	      } catch (InvocationTargetException e) {
	        throw new IllegalStateException(CANNOT_CONSTRUCT_CONVERTER, e);
	      } catch (NoSuchMethodException e) {
	        throw new IllegalStateException(CANNOT_CONSTRUCT_CONVERTER, e);
	      }
	    }
	  }

	
	  private static List<String> parseCsvLine(String line) {
		    try {
		      return Arrays.asList(CSVUtils.parseLine(line));
			   }
			   catch (IOException e) {
		      List<String> list = Lists.newArrayList();
		      list.add(line);
		      return list;
		   	}
		  }

	
	  static BufferedReader open(String inputFile) throws IOException {
		    InputStream in;
		    try {
		      in = Resources.getResource(inputFile).openStream();
		    } catch (IllegalArgumentException e) {
		      in = new FileInputStream(new File(inputFile));
		    }
		    return new BufferedReader(new InputStreamReader(in, Charsets.UTF_8));
		  }

	  public static LogisticModelParameters loadFrom(File in) throws IOException {
		    InputStream input = new FileInputStream(in);
		    try {
		      return loadFrom(input);
		    } finally {
		      Closeables.close(input, true);
		    }
		  }
	  
	  public static LogisticModelParameters loadFrom(InputStream in) throws IOException {
		    LogisticModelParameters result = new LogisticModelParameters();
		    result.readFields(new DataInputStream(in));
		    return result;
		  }
/*	  public void readFields(DataInput in) throws IOException {
		    targetVariable = in.readUTF();
		    int typeMapSize = in.readInt();
		    typeMap = Maps.newHashMapWithExpectedSize(typeMapSize);
		    for (int i = 0; i < typeMapSize; i++) {
		      String key = in.readUTF();
		      String value = in.readUTF();
		      typeMap.put(key, value);
		    }
		    numFeatures = in.readInt();
		    useBias = in.readBoolean();
		    maxTargetCategories = in.readInt();
		    int targetCategoriesSize = in.readInt();
		    targetCategories = Lists.newArrayListWithCapacity(targetCategoriesSize);
		    for (int i = 0; i < targetCategoriesSize; i++) {
		      targetCategories.add(in.readUTF());
		    }
		    lambda = in.readDouble();
		    learningRate = in.readDouble();
		    csv = null;
		    lr = new OnlineLogisticRegression();
		    lr.readFields(in);
		  }*/
	  public static void main(String args[]) throws IOException
	  {
		  File inputFile = new File(modelFile);
		  InputStream input = new FileInputStream(inputFile);
		  try {
		      DataInput in=new DataInputStream(input);
			   String targetVariable = in.readUTF();
			   int typeMapSize = in.readInt();
			   typeMap = Maps.newHashMapWithExpectedSize(typeMapSize);
			   for (int i = 0; i < typeMapSize; i++) {
				      String key = in.readUTF();
				      String value = in.readUTF();
				      typeMap.put(key, value);
				    }
			    int numFeatures = in.readInt();
			    boolean includeBiasTerm = in.readBoolean();
			    int maxTargetCategories = in.readInt();
			    int targetCategoriesSize = in.readInt();
			    ArrayList targetCategories = Lists.newArrayListWithCapacity(targetCategoriesSize);
			    for (int i = 0; i < targetCategoriesSize; i++) {
			      targetCategories.add(in.readUTF());
			    }
			    double lambda = in.readDouble();
			    double learningRate = in.readDouble();
			    OnlineLogisticRegression lr = new OnlineLogisticRegression();
			    System.out.println("Started reading fields");
			    lr.readFields(in);
			    lr =new OnlineLogisticRegression(maxTargetCategories,numFeatures,new L1()).lambda(lambda).learningRate(learningRate).alpha(1 - 1.0e-3); 
			    BufferedReader predLine = OnlineLogisticRegressionPredict.open(inputFileName);
			    String line = predLine.readLine();
			    firstLine(line);
			    System.out.println(line);
			    line = predLine.readLine();
			    System.out.println(line);
			    PrintWriter output=new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);
			    while (line != null) {
		            Vector v = new SequentialAccessSparseVector(numFeatures);
		            List<String> values = parseCsvLine(line);
		            for (Integer predictor : predictors) {
		                String value;
		                if (predictor >= 0) {
		                  value = values.get(predictor);
		                } else {
		                  value = null;
		                }
		                predictorEncoders.get(predictor).addToVector(value, v);
		              }
		            System.out.println(v);
		            Vector scoreVec = lr.classify(v);
		            System.out.println(scoreVec);
		            double score = lr.classifyScalar(v);
		            output.printf(Locale.ENGLISH, "%.3f%n",  score);
		            line = predLine.readLine();
			    }

			   
			    
			    
		    } finally {
		      Closeables.close(input, true);
		    }
		  
		  
		  
	  }
}
