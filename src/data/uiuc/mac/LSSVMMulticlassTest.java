package data.uiuc.mac;

import java.io.BufferedWriter;
import java.io.File;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import latent.LatentRepresentation;
import latent.lssvm.multiclass.LSSVMMulticlassFastBagMIL;
import latent.variable.BagMIL;

import struct.STrainingSample;
import data.io.BagReader;
import fr.lip6.jkernelmachines.type.TrainingSample;

public class LSSVMMulticlassTest {
	

	
	private static int cpmax = 500;
	private static int cpmin = 10;
//	private static int init = 0;
	private static int optim = 1;
//	private static double epsilon = 1e-2;
	
	//big path
	public static String simDir = "/home/wangxin/results/gaze_voc_actions_stefan/stdlssvm/";
	public static String sourceDir = "/home/wangxin/Data/gaze_voc_actions_stefan/";
	
//	public static String simDir = "/home/wangxin/Data/ferrari_data/reduit_singlebb/";
//	public static String sourceDir = "/home/wangxin/Data/ferrari_data/POETdataset/POETdataset/";

	
	public static int split = 1;
	public static int scale = 100;
	//ensure dimension of features
	private static int numWords = 2048;

	public static void main(String[] args) {
		
//	    double[] lambdaCV = {1e-4,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,1e-3};//1e-4
	    double[] lambdaCV = {1e-3};//1e-4
	    double[] epsilonCV = {1e-2};//1e-2
	    

	    String[] classes = {args[0]};	    
	    int[] scaleCV = {Integer.valueOf(args[1])};
	    //int[] scaleCV = {50};
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
				String classifierDir = simDir + "classifier/lssvm/" ;
				//example_files
				String inputDir = sourceDir + "example_files/"+scale;
				
				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);
    			
    			for(double epsilon : epsilonCV) {
    		    	for(double lambda : lambdaCV) {
		    			
    		    		LSSVMMulticlassFastBagMIL lsvm = new LSSVMMulticlassFastBagMIL(); 
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
					List<TrainingSample<BagMIL>> listTrain = BagReader.readBagMIL(inputDir + "/"+className+"_train_scale_"+scale+"_matconvnet_m_2048_layer_20.txt", numWords);
					
					List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleTrain = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();
					for(int i=0; i<listTrain.size(); i++) {
						exampleTrain.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listTrain.get(i).sample,0), listTrain.get(i).label));
					}

					List<TrainingSample<BagMIL>> listTest = BagReader.readBagMIL(inputDir + "/"+className+"_val_scale_"+scale+"_matconvnet_m_2048_layer_20.txt", numWords);
					List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleTest = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();
					for(int i=0; i<listTest.size(); i++) {
						exampleTest.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listTest.get(i).sample,0), listTest.get(i).label));
					}
			
	    			for(double epsilon : epsilonCV) {
	    		    	for(double lambda : lambdaCV) {
			    			
	    		    		LSSVMMulticlassFastBagMIL lsvm = new LSSVMMulticlassFastBagMIL(); 
							lsvm.setLambda(lambda);
							//lsvm.setInitType(init);
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
								System.err.println("train - " + cls + "\tscale= " + scale + "\tap= " + ap_train + "\tlambda= " + lambda + "\tepsilon= " + epsilon);
								
//								acc = lsvm.test(exampleTest);
//								
//								fileClassifier = new File(classifierDir + "/" + className + "/" + className + "_" + scale + suffix + "_acc_" + acc + ".ser");
//								fileClassifier.getAbsoluteFile().getParentFile().mkdirs();
//								System.out.println("save classifier " + fileClassifier.getAbsolutePath());
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
//									lsvm = (LSSVMMulticlassFastBagMIL) ois.readObject();
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
								double ap = lsvm.testAPRegion(exampleTest, epsilon, lambda,scale, simDir, className);
								File resFile=new File(simDir+"res_lssvm.txt");
								try {
									BufferedWriter out = new BufferedWriter(new FileWriter(resFile, true));
									out.write(className+" "+"notradeoff"+" "+scale+" "+lambda+" "+epsilon+" "+ap+" "+ap_train+"\n");
									out.flush();
									out.close();
									
								} catch (IOException e) {
									// TODO Auto-generated catch block
									e.printStackTrace();
								}
								System.err.println("test - " + cls + "\tscale= " + scale + "\tap= " + ap + "\tlambda= " + lambda + "\tepsilon= " + epsilon);
								System.out.println("\n");
							}
		    			}
		    		}
	    		}

    		
    		}
	    }
	}}
	
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