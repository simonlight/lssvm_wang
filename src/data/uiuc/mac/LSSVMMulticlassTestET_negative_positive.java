package data.uiuc.mac;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import latent.LatentRepresentation;
//import latent.lssvm.multiclass.LSSVMMulticlassFastBagMILET;
import latent.lssvm.multiclass.LSSVMMulticlassFastBagMILET_negative_positive;
import latent.variable.BagMIL;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import struct.STrainingSample;
import data.io.BagReader;
import fr.lip6.jkernelmachines.type.TrainingSample;

public class LSSVMMulticlassTestET_negative_positive {
	
	static Option cOption = OptionBuilder.withArgName("regularization parameter C")
			.hasArg()
			.withDescription("c value")
			.withLongOpt("c")
			.create("c");
	static Option initOption = OptionBuilder.withArgName("init type")
			.hasArg()
			.withDescription("init")
			.withLongOpt("init")
			.create("i");
	static Option optimOption = OptionBuilder.withArgName("optim")
			.hasArg()
			.withDescription("optim")
			.withLongOpt("optim")
			.create("o");
	static Option cpmaxOption = OptionBuilder.withArgName("cutting plane")
			.hasArg()
			.withDescription("maximum number of cutting plane")
			.withLongOpt("cuttingPlaneMax")
			.create("cpmax");
	static Option cpminOption = OptionBuilder.withArgName("cutting plane")
			.hasArg()
			.withDescription("minimum number of cutting plane")
			.withLongOpt("cuttingPlaneMax")
			.create("cpmin");
	static Option epsilonOption = OptionBuilder.withArgName("epsilon")
			.hasArg()
			.withDescription("epsilon")
			.withLongOpt("epsilon")
			.create("eps");
	
	static Option scaleOption = OptionBuilder.withArgName("scale")
			.hasArg()
			.withDescription("scale")
			.withLongOpt("scale")
			.create("s");
	static Option splitOption = OptionBuilder.withArgName("slit")
			.hasArg()
			.withDescription("split")
			.withLongOpt("split")
			.create("sp");
	static Option numWordsOption = OptionBuilder.withArgName("numWords")
			.hasArg()
			.withDescription("numWords")
			.withLongOpt("numWords")
			.create("w");
	
	static Options options = new Options();
	
	static {
		options.addOption(cOption);
		options.addOption(initOption);
		options.addOption(optimOption);
		options.addOption(cpmaxOption);
		options.addOption(cpminOption);
		options.addOption(epsilonOption);
		options.addOption(scaleOption);
		options.addOption(splitOption);
		options.addOption(numWordsOption);
	}
	
	private static int cpmax = 500;
	private static int cpmin = 10;
	private static double lambda = 1e-4;
	private static int init = 0;
	private static int optim = 1;
	private static double epsilon = 1e-2;
	//racine
	public static String simDir = "/home/wangxin/Data/ferrari_data/reduit_allbb/";
	public static String sourceDir = "/home/wangxin/Data/ferrari_data/POETdataset/POETdataset/";
	
	
	public static int split = 1;
	public static int scale = 100;
	//ensure dimension of features
	private static int numWords = 2048;

	public static void main(String[] args) {
		
		// Option parsing		
	    // Create the parser
	    CommandLineParser parser = new GnuParser();
	    try {
	    	// parse the command line arguments
	    	CommandLine line = parser.parse( options, args );

	    	if(line.hasOption("init")) {
	    		init = Integer.parseInt(line.getOptionValue("i"));
	    	}
	    	if(line.hasOption("optim")) {
	    		optim = Integer.parseInt(line.getOptionValue("o"));
	    	}
	    	if(line.hasOption("cuttingPlaneMax")) {
	    		cpmax = Integer.parseInt(line.getOptionValue("cpmax"));
	    	}
	    	if(line.hasOption("cuttingPlaneMin")) {
	    		cpmin = Integer.parseInt(line.getOptionValue("cpmin"));
	    	}
	    	
	    	if(line.hasOption("numWords")) {
	    		numWords = Integer.parseInt(line.getOptionValue("w"));
	    	}
	    	
	    }
	    catch(ParseException exp) {
	        // oops, something went wrong
	        System.err.println( "Parsing failed.  Reason: " + exp.getMessage() );
        	HelpFormatter formatter = new HelpFormatter();
        	formatter.printHelp( "Parameters", options );
        	System.exit(-1);
	    }
		
	    double[] lambdaCV = {1e-3};
	    double[] epsilonCV = {1e-2};
	    double[] tradeoffCV = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
//	    String[] classes = {"cat", "dog", "bicycle", "motorbike", "boat", "aeroplane", "horse", "cow", "sofa", "diningtable"};
	    
//	    String[] classes = {"cat", "bicycle", "motorbike", "boat", "aeroplane"};
//	    String[] classes = {"dog", "horse", "cow", "sofa", "diningtable"};
	    
//	    String[] classes = {"cat", "dog"};
//	    String[] classes = {"bicycle", "motorbike",};
//	    String[] classes = {"boat", "aeroplane"};
//	    String[] classes = {"horse", "cow"};
//	    String[] classes = {"sofa", "diningtable"};
	    
	    String[] classes = {args[0]};	    
//	    String[] classes = {"cat"};
//	    String[] classes = {"dog"};
//	    String[] classes = {"bicycle"};
//	    String[] classes = {"motorbike"};
//	    String[] classes = {"boat"};
//	    String[] classes = {"aeroplane"};
	    // go horse right now, nothing done
//	    String[] classes = {"horse"};
//	    String[] classes = {"cow"};
//	    String[] classes = {"sofa"};
//	    String[] classes = {"diningtable"};
	    //String[] classes = { "motorbike", "boat", "aeroplane", "horse", "cow", "sofa", "diningtable"};
	    //String[] classes = { "cat"};
	    int[] scaleCV = {Integer.valueOf(args[1])};
//	    int[] scaleCV = {90};
//	    int[] scaleCV = {80};
//	    int[] scaleCV = {70};
//      int[] scaleCV = {60};
//	    int[] scaleCV = {50};
	    
	    //int[] splitCV = {1,2,3,4,5};
	    int[] splitCV = {1};
	    
	    System.out.println("lambda " + Arrays.toString(lambdaCV));
	    System.out.println("epsilon " + Arrays.toString(epsilonCV));
	    System.out.println("scale " + Arrays.toString(scaleCV));
	    System.out.println("split " + Arrays.toString(splitCV) + "\n");
		
	    boolean compute = false;
	    String features = "pure";
	    
	    
	    for(String className: classes){
		for(int scale : scaleCV) {
    		for(int split : splitCV) {
    			
    			String cls = String.valueOf(split);
    			//sauvgarder les classifieurs
				String classifierDir = simDir + "classifier/lssvm_et/" ;
				//example_files
				String inputDir = sourceDir + "example_files/"+scale;

				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);
    			
    			for(double epsilon : epsilonCV) {
    		    	for(double lambda : lambdaCV) {
		    			
    		    		LSSVMMulticlassFastBagMILET_negative_positive lsvm = new LSSVMMulticlassFastBagMILET_negative_positive(); 
						lsvm.setLambda(lambda);
						//lsvm.setInitType(init);
						lsvm.setOptim(optim);
						lsvm.setCpmax(cpmax);
						lsvm.setCpmin(cpmin);
						lsvm.setEpsilon(epsilon);
						
		    			String suffix = "_" + lsvm.toString();
		    			System.out.println(suffix);
		    			File fileClassifier = testPresenceFile(classifierDir + "/" + className + "/", className + "_" + scale + suffix);
		    			if(fileClassifier == null) {
		    				compute = true;
		    			}
    		    	}
    			}
				
//				if(compute) {
				if(true) {
					//
					List<TrainingSample<BagMIL>> listTrain = BagReader.readBagMIL(inputDir + "/"+className+"_train_scale_"+scale+"_matconvnet_m_2048_layer_20.txt", numWords);
					
					//List<TrainingSample<BagMIL>> listTrain = BagReader.readBagMIL(inputDir + "/multiclass_" + features + "_train_scale_" + scale + ".txt", numWords);
					List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleTrain = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();
					for(int i=0; i<listTrain.size(); i++) {
//						for(int i=0; i<10; i++) {
						exampleTrain.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listTrain.get(i).sample,0), listTrain.get(i).label));
					}

					List<TrainingSample<BagMIL>> listTest = BagReader.readBagMIL(inputDir + "/"+className+"_val_scale_"+scale+"_matconvnet_m_2048_layer_20.txt", numWords);
					List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleTest = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();
					for(int i=0; i<listTest.size(); i++) {
						exampleTest.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listTest.get(i).sample,0), listTest.get(i).label));
					}
			
	    			for(double epsilon : epsilonCV) {
	    		    	for(double lambda : lambdaCV) {
	    		    		for(double tradeoff : tradeoffCV){
		    		    		LSSVMMulticlassFastBagMILET_negative_positive lsvm = new LSSVMMulticlassFastBagMILET_negative_positive(); 
								lsvm.setObjectClass(className);
		    		    		lsvm.setLambda(lambda);
		    		    		lsvm.setLossDict("/home/wangxin/Data/ferrari_data/reduit_allbb/ETLoss_dict/ETLOSS+_"+scale+".loss", "/home/wangxin/Data/ferrari_data/reduit_allbb/ETLoss_dict/ETLOSS-_"+className+"_"+scale+".loss");
								lsvm.setTradeOff(tradeoff);
		    		    		lsvm.setOptim(optim);
								lsvm.setCpmax(cpmax);
								lsvm.setCpmin(cpmin);
								lsvm.setEpsilon(epsilon);
								
								String suffix = "_" + lsvm.toString();
								File fileClassifier = testPresenceFile(classifierDir + "/" + className + "/", className + "_" + scale + suffix);
				    			//if(compute || fileClassifier == null) {
				    			if(true){
				    				lsvm.train(exampleTrain);
				    				double ap_train = lsvm.testAP(exampleTrain);
									System.err.println("train " + String.valueOf(tradeoff)+"\t"+cls + "\tscale= " + scale + "\tap= " + ap_train + "\tlambda= " + lambda + "\tepsilon= " + epsilon);
									
									//double acc = lsvm.test(exampleTrain);
									//System.err.println("train - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda + "\tepsilon= " + epsilon);
									
									//acc = lsvm.test(exampleTest);
									
									//fileClassifier = new File(classifierDir + "/" + className + "/" + className + "_" + scale + suffix + "_acc_" + acc + ".ser");
									//fileClassifier.getAbsoluteFile().getParentFile().mkdirs();
									//System.out.println("save classifier " + fileClassifier.getAbsolutePath());
									// save classifier
	//								ObjectOutputStream oos = null;
	//								try {
	//									oos = new ObjectOutputStream(new FileOutputStream(fileClassifier.getAbsolutePath()));
	//									oos.writeObject(lsvm);
	//								} 
	//								catch (FileNotFoundException e) {
	//									e.printStackTrace();
	//								} 
	//								catch (IOException e) {
	//									e.printStackTrace();
	//								}
	//								finally {
	//									try {
	//										if(oos != null) {
	//											oos.flush();
	//											oos.close();
	//										}
	//									} catch (IOException e) {
	//										e.printStackTrace();
	//									}
	//								}
	//								
	//								// load classifier
	//								ObjectInputStream ois;
	//								System.out.println("read classifier " + fileClassifier.getAbsolutePath());
	//								try {
	//									ois = new ObjectInputStream(new FileInputStream(fileClassifier.getAbsolutePath()));
	//									lsvm = (LSSVMMulticlassFastBagMILET) ois.readObject();
	//									lsvm.showParameters();
	//								} catch (FileNotFoundException e) {
	//									// TODO Auto-generated catch block
	//									e.printStackTrace();
	//								} catch (IOException e) {
	//									// TODO Auto-generated catch block
	//									e.printStackTrace();
	//								} catch (ClassNotFoundException e) {
	//									// TODO Auto-generated catch block
	//									e.printStackTrace();
	//								}
									double ap = lsvm.testAPRegion(exampleTest, scale, simDir, className, tradeoff);
									File resFile=new File(simDir+"res_neg_pos_no_prediction.txt");
									try {
										BufferedWriter out = new BufferedWriter(new FileWriter(resFile, true));
										//out.write(className+" "+scale+" "+acc+" "+ap+"\n");
										out.write(className+" "+String.valueOf(tradeoff)+" "+scale+" "+" "+ap+" "+ap_train+"\n");
										out.flush();
										out.close();
										
									} catch (IOException e) {
										// TODO Auto-generated catch block
										e.printStackTrace();
									}
									System.err.println("test - " + cls + "\tscale= " + scale + "\tap= " + ap + "\tlambda= " + lambda + "\tepsilon= " + epsilon);
	//								System.out.println("\n");
								}
			    			}
	    		    	}
	    			
		    		}
	    		}
	    	}
	    }
	}
	    }
	
	public static File testPresenceFile(String dir, String test) {
		boolean testPresence = false;
		File classifierDir = new File(dir);
		File file = null;
		if(classifierDir.exists()) {
			String[] f = classifierDir.list();
			//System.out.println(Arrays.toString(f));
			
			for(String s : f) {
				if(s.contains(test)) {
					testPresence = true;
					file = new File(dir + "/" + s);
				}
			}
		}
		System.out.println("presence " + testPresence + "\t" + dir + "\t" + test + "\tfile " + (file == null ? null : file.getAbsolutePath()));
		return file;
	}
	
}
