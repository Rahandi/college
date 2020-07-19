package youtube_downloader;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class ExtractData {
	JSONObject obj = null;
	
	public ExtractData(JSONObject obje) {
		obj = obje;
	}
	
	public String cekData() throws JSONException {
		String error = obj.getString("error");
		if(error.equals("null")) {
			obj = obj.getJSONObject("result");
		}
		return error;
	}
	
	public String getTitle() throws JSONException {
		String title = obj.getString("title");
		return title;
	}
	
	public String getThumbnail() throws MalformedURLException, IOException, JSONException {
		InputStream gambar = new URL(obj.getString("thumbnail")).openStream();
		String path = "thumbnail.jpg";
		File paths = new File(path);
		paths.delete();
		Files.copy(gambar, Paths.get(path));
		paths = new File(path);
		paths.deleteOnExit();
		return path;
	}
	
	public JSONArray getDownload() throws JSONException {
		JSONArray result = obj.getJSONArray("videolist");
		return result;
	}
}
