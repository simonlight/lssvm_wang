package io;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;


public class LossDictPositive {

	static HashMap<String , Double> map = new HashMap<String , Double>(); 
	
    public static void traverse(File parentNode) {

			try {
				File childNodes[] = parentNode.listFiles();
				for (File childNode : childNodes) {
					FileReader fr = new FileReader(childNode.getAbsolutePath());
					BufferedReader br=new BufferedReader(fr);
					double ETLoss = Double.parseDouble(br.readLine());
					br.close();
					fr.close();
					map.put(childNode.getName(), ETLoss); 
				
				}
	        			
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (NumberFormatException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			
			
        }
    }
	public static int convert(int scale){
		return (int)(100-Math.sqrt(scale)*10 + 10);
	}
	public static void main(String[] args) throws IOException, ClassNotFoundException {
////		String classes = "cat";
////	    String classes = "dog";
////	    String classes = "bicycle";
////	    String classes = "motorbike";
////	    String classes = "boat";
////	    String classes = "aeroplane";
//	    // go horse right now, nothing done
//	    String classes = "horse";
////	    String classes = "cow";
////  String classes = "sofa";
////	    String classes = "diningtable";
//	    int[] scale_list={25};
//		String[] classList={"cat", "dog", "bicycle", "motorbike", "boat", "aeroplane", "horse", "cow", "sofa", "diningtable"};
//		for (int scale_index=0; scale_index<scale_list.length;scale_index++){
//			int scale=scale_list[scale_index];
//			System.out.println(scale);
//			for (String imClass:classList){
//				File inputFolder = new File("/home/wangxin/Data/ferrari_data/reduit_allbb/negative_ETLoss_ratio/"+classes+"/"+imClass+"/"+String.valueOf(scale)+"/");
//	        	traverse(inputFolder);
//	        	System.out.println(map.size());
//				
//	        }
//        	ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream("/home/wangxin/Data/ferrari_data/reduit_allbb/ETLoss_dict/ETLOSS-_"+classes+"_"+convert(scale)+".loss"));  
//        	os.writeObject(map);
//        	os.close();
//        	map.clear();
//        }
		
//		String[] classList={"dog","cat", "motorbike", "boat", "aeroplane", "horse", "cow", "sofa", "diningtable", "bicycle"};
		String[] classList={"jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"};
//		int[] scale_list={1,4,9};
//		int[] scale_list={16};
//		int[] scale_list={25};
		int[] scale_list={36};
	    for (int scale_index=0; scale_index<scale_list.length;scale_index++){
        	int scale=scale_list[scale_index];
        	System.out.println(scale);
        	for (String imClass:classList){
	        	File inputFolder = new File("/home/wangxin/Data/gaze_voc_actions_stefan/ETLoss_ratio/"+imClass+"/"+String.valueOf(scale)+"/");
//	        	File inputFolder = new File("/home/wangxin/Data/ferrari_data/POETdataset/POETdataset/ETLoss_ratio/"+imClass+"/"+String.valueOf(scale)+"/");
	        	traverse(inputFolder);
        	}
//        	ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream("/local/wangxin/Data/ferrari_data/reduit_allbb/ETLoss_dict/ETLOSS+_"+convert(scale)+".loss"));  
        	ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream("/home/wangxin/Data/ferrari_data/POETdataset/POETdataset/ETLoss_dict/ETLOSS+_"+convert(scale)+".loss"));  
        	os.writeObject(map);
        	os.close();
        	map.clear();
		
        }
		
//
//	            ObjectInputStream is = new ObjectInputStream(new FileInputStream(  
//                "/home/wangxin/Data/ferrari_data/reduit_allbb/ETLoss_dict/ETLOSS+_"+"100"+".loss"));  
//        HashMap<String[] , Double>  temp = (HashMap<String[], Double> ) is.readObject();// 从流中读取User的数据  
//        System.out.println(temp.size());
//        is.close();
        
    } 
		 
	}


