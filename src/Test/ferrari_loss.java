package Test;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Set;

public class ferrari_loss {

	public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
		// TODO Auto-generated method stub
		String filename = "/local/wangxin/Data/ferrari_gaze/ETLoss_dict/ETLOSS+_50.loss";
	    ObjectInputStream is = new ObjectInputStream(new FileInputStream(filename));  
        HashMap<String , Double>  temp = (HashMap<String, Double> ) is.readObject();// 从流中读取User的数据  
        is.close();
        String query = "horse_2010_000413";
        for (int x = 0; x<6;x++){
            for (int y = 0; y<6;y++){
        System.out.println(query+"_"+String.valueOf(x)+"_"+String.valueOf(y)+".txt:"+temp.get(query+"_"+String.valueOf(x)+"_"+String.valueOf(y)+".txt"));
            }}
//        ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(filename));  
//    	os.writeObject(temp);
//    	os.close();
	}

}
