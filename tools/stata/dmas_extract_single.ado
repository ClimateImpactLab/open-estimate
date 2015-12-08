program dmas_extract_single

version 13.1
args apikey coeff infoid id

if ("`id'" == "") {
    disp as txt "id not provided; using last result"
    local id = "$DMAS_LAST_RESULT"
}

local dmas_urlstr = "extract_stata_single?apikey=`apikey'&coeff=`coeff'&infoid=`infoid'&id=`id'&ts=$S_TIME"

dmas_get_api "`dmas_urlstr'", as_model(1)

end

