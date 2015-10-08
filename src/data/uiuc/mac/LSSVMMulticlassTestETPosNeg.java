package data.uiuc.mac;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import latent.LatentRepresentation;
import latent.lssvm.multiclass.LSSVMMulticlassFastBagMILETPosNeg;
import latent.variable.BagMIL;
import struct.STrainingSample;
import data.io.BagReader;
import fr.lip6.jkernelmachines.type.TrainingSample;

public class LSSVMMulticlassTestETPosNeg {
	
	
	
	public static void main(String[] args) {
		
		String sourceDir = "/home/wangxin/Data/gaze_voc_actions_stefan/";
		String simDir = "/home/wangxin/results/gaze_voc_actions_stefan/ETPosNeg/";
//		String testResultFileName = "test.txt";
//		String sourceDir = "/local/wangxin/Data/gaze_voc_actions_stefan/";
//		String simDir = "/local/wangxin/results/gaze_voc_actions_stefan/ETPosNeg/";
		String testResultFileName = "resposneg.txt";
		String detailFolder= "overlappingC1e-3";

		//	public static String simDir = "/home/wangxin/Data/ferrari_data/reduit_singlebb/";
		//	public static String sourceDir = "/home/wangxin/Data/ferrari_data/POETdataset/POETdataset/";
		String lossPath = sourceDir+"ETLoss_dict/";
		String lossPathNeg = sourceDir+"ETLoss_dict_context/";
		
		//ensure dimension of features
		int numWords = 2048;
		
		int optim = 1;
		int epochsLatentMax = 50;
		int epochsLatentMin = 5;
		int cpmax = 500;
		int cpmin = 10;
		
	    double[] lambdaCV = {1e-3};
//	    double[] lambdaCV = {1e-4};
	    double[] epsilonCV = {1e-2};

	    double[] tradeoffCV = {0.5};
//	    double[] tradeoffCV = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	    String[] classes = {args[0]};
//	    String[] classes = {"walking"};
	    int[] scaleCV = {Integer.valueOf(args[1])};
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
    			
//    			for(double epsilon : epsilonCV) {
//    		    	for(double lambda : lambdaCV) {
//		    			
//    		    		LSSVMMulticlassFastBagMILET lsvm = new LSSVMMulticlassFastBagMILET();
//    		    		
//    		    		lsvm.setOptim(optim);
//    		    		lsvm.setEpochsLatentMax(epochsLatentMax);
//    		    		lsvm.setEpochsLatentMin(epochsLatentMin);
//    		    		lsvm.setCpmax(cpmax);
//    		    		lsvm.setCpmin(cpmin);
//    		    		lsvm.setLambda(lambda);
//    		    		lsvm.setEpsilon(epsilon);
//						lsvm.setLossDict(lossPath+"ETLOSS+_"+scale+".loss");
//						lsvm.setTradeOff(tradeoff);
//						
//		    			String suffix = "_" + lsvm.toString();
//		    			System.out.println(suffix);
//		    			File fileClassifier = testPresenceFile(classifierDir + "/" + className + "/", className + "_" + scale + suffix);
//		    			if(fileClassifier == null) {
//		    				compute = true;
//		    			}
//    		    	}
//    			}
				
//				if(compute) {
				if(true) {
					//
					List<TrainingSample<BagMIL>> listTrain = BagReader.readBagMIL(inputDir + "/"+className+"_train_scale_"+scale+"_matconvnet_m_2048_layer_20.txt", numWords);
					List<TrainingSample<BagMIL>> listTest = BagReader.readBagMIL(inputDir + "/"+className+"_val_scale_"+scale+"_matconvnet_m_2048_layer_20.txt", numWords);
			
	    			for(double epsilon : epsilonCV) {
	    		    	for(double lambda : lambdaCV) {
	    		    		for(double tradeoff : tradeoffCV){
    						
	    		    		List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleTrain = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();
    						for(int i=0; i<listTrain.size(); i++) {
    							exampleTrain.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listTrain.get(i).sample,0), listTrain.get(i).label));
    						}

    						List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleTest = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();
    						for(int i=0; i<listTest.size(); i++) {
    							exampleTest.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listTest.get(i).sample,0), listTest.get(i).label));
    						}

	    		    		LSSVMMulticlassFastBagMILETPosNeg lsvm = new LSSVMMulticlassFastBagMILETPosNeg(); 
							
	    		    		lsvm.setOptim(optim);
	    		    		lsvm.setEpochsLatentMax(epochsLatentMax);
	    		    		lsvm.setEpochsLatentMin(epochsLatentMin);
	    		    		lsvm.setCpmax(cpmax);
	    		    		lsvm.setCpmin(cpmin);
	    		    		lsvm.setLambda(lambda);
	    		    		lsvm.setEpsilon(epsilon);
							
	    		    		lsvm.setLossDict(lossPath+"ETLOSS+_"+scale+".loss");
	    		    		lsvm.setNegLossDict(lossPathNeg+"ETLOSS-_"+scale+".loss");
	    		    		lsvm.setTradeOff(tradeoff);
							//
							String suffix = "_" + lsvm.toString();
							File fileClassifier = testPresenceFile(classifierDir + "/" + className + "/", className + "_" + scale + suffix);
			    			//if(compute || fileClassifier == null) {
			    			if(true){
			    				lsvm.train(exampleTrain);
			    				double ap_train = lsvm.testAP(exampleTrain);
			    				
			    				System.err.println("train " + String.valueOf(tradeoff)+" "+cls + " scale= " + scale + " ap= " + ap_train + " lambda= " + lambda + " epsilon= " + epsilon);
								
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
			    				
			    				double ap = lsvm.testAPRegion(exampleTest, epsilon, lambda,scale, simDir, className, tradeoff,detailFolder);
								File resFile=new File(simDir+testResultFileName);
								try {
									BufferedWriter out = new BufferedWriter(new FileWriter(resFile, true));
									//out.write(className+" "+scale+" "+acc+" "+ap+"\n");
									out.write(className+" "+String.valueOf(tradeoff)+" "+scale+" "+lambda+" "+epsilon+" "+ap+" "+ap_train+"\n");
									out.flush();
									out.close();
									
								} catch (IOException e) {
									// TODO Auto-generated catch block
									e.printStackTrace();
								}
								System.err.println("test "+ String.valueOf(tradeoff)+" "+cls + " scale= " + scale + " ap= " + ap + " lambda= " + lambda + " epsilon= " + epsilon);
								System.out.println("\n");
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