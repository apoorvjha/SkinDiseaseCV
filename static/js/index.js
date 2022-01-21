function validate(){
	var fInput=document.getElementById("profilePic");
	var email=document.getElementById("email");
	var userId=document.getElementById("userId");
	var password=document.getElementById("password");
	var flag=0;
	if(fInput.value.length!=0){
		/* file validation */
		var validExt= /(\.jpg|\.jpeg|\.png|\.gif)$/i;
		if (!validExt.exec(fInput.value)) {
			document.getElementById("filecheck").innerHTML='<font color="red">Only JPEG, JPG, PNG and GIF formats are supported!</font>';
            fInput.style.borderColor="red";
			flag=0;
		}
		else{
			document.getElementById("filecheck").innerHTML='<font color="green">Looks good!</font>';
			flag=1;
            fInput.style.borderColor="green";
		} 
    }
	if(email.value.length!=0){
		/* email validation */
		var res=email.value.split('@');
		var ext = /(\.com|\.in|\.ac.in|\.net)$/i;
		if (res[0].length<2 || res[1].length<4 || !ext.exec(res[1]) || res[1]==undefined || res[0]==undefined){
			document.getElementById("emailcheck").innerHTML='<font color="red">Please enter a valid email!</font>';
			flag=0;
            email.style.borderColor="red";
		}
		else{
			document.getElementById("emailcheck").innerHTML='<font color="green">Looks good!</font>';
			flag=1;
            email.style.borderColor="green";
		} 
    }

	if(userId.value.length!=0){
		/* userId validation */
		var res=userId.value
		if (res.length<5 || res.length>20) {
			document.getElementById("idcheck").innerHTML='<font color="red">Length of User Id should be in between 5-20 characters!</font>';
            userId.style.borderColor="red";
			flag=0;
		}
		else{
			document.getElementById("idcheck").innerHTML='<font color="green">Looks good!</font>';
            userId.style.borderColor="green";
			flag=1;
		} 
    }

	if(password.value.length!=0){
		/* password validation */
		var res=password.value
		if (res.length<6 || res.length>15) {
			document.getElementById("passcheck").innerHTML='<font color="red">length of Password should be in between 6-15 characters!</font>';
            password.style.borderColor="red";
			flag=0;
		}
		else{
			document.getElementById("passcheck").innerHTML='<font color="green">Looks good!</font>';
            password.style.borderColor="green";
			flag=1;
		}
	 
    }

    if(flag==1){
        return true;
    }
    else{
        return false;
    }	
}

function validate_login(){
	var userId=document.getElementById("userId");
	var password=document.getElementById("password");
	var flag=0;

	if(userId.value.length!=0){
		/* userId validation */
		var res=userId.value
		if (res.length<5 || res.length>20) {
			document.getElementById("idcheck").innerHTML='<font color="red">&nbsp;&nbsp;&nbsp;&nbsp;Length of User Id should be in between 5-20 characters!</font>';
            userId.style.borderColor="red";
			flag=0;
		}
		else{
			document.getElementById("idcheck").innerHTML='<font color="green">&nbsp;&nbsp;&nbsp;&nbsp;Looks good!</font>';
			flag=1;
            userId.style.borderColor="green";
		} 
    }

	if(password.value.length!=0){
		/* password validation */
		var res=password.value
		if (res.length<6 || res.length>15) {
			document.getElementById("passcheck").innerHTML='<font color="red">&nbsp;&nbsp;&nbsp;&nbsp;Length of Password should be in between 6-15 characters!</font>';
            password.style.borderColor="red";
			flag=0;
		}
		else{
			document.getElementById("passcheck").innerHTML='<font color="green">&nbsp;&nbsp;&nbsp;&nbsp;Looks good!</font>';
			flag=1; 
            password.style.borderColor="green";
		}
	 
    }

    if(flag==1){
        return true;
    }
    else{
        return false;
    }	
}

function remove_msg(){
	if(document.getElementById("alert alert-success")!=null){
		document.getElementById("alert alert-success").style.display='none';
	}
	if(document.getElementById("alert alert-danger")!=null){
		document.getElementById("alert alert-danger").style.display='none';
	}
}